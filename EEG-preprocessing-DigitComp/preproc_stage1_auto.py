"""
Stage 1 — Automated preprocessing

Runs all subjects and produces:
  preprocessing/stage1/{subj}_raw.fif         band-pass filtered raw (no reference yet)
  preprocessing/stage1/{subj}_hpfilt_raw.fif  HP-only copy for bad-channel inspection
  preprocessing/bridging/{subj}_bridging.png
  preprocessing/bridging/{subj}_bridged_pairs.npy

REST re-referencing is deferred to stage 3, after bad channels are known.

Run:
    python stage1_auto.py

Subjects that are already complete are skipped automatically, so the script
is safe to re-run after interruption.
"""

import os
import re
import json

import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = "" # add project path here
RAW    = os.path.join(BASE, "raw_data")
OUT1   = os.path.join(BASE, "preprocessing", "stage1")
BRIDGE = os.path.join(BASE, "preprocessing", "bridging")
MANUAL = os.path.join(BASE, "preprocessing", "manual")

# ── Constants ──────────────────────────────────────────────────────────────
# p1 and p2 were recorded at a different sampling rate and must be resampled.
RESAMPLE_SUBJECTS = {"p1", "p2"}
TARGET_SFREQ      = 512

STIM_CODES = np.array([51, 53])
RESP_CODES = np.array([55, 56])

# ── Subject files ──────────────────────────────────────────────────────
bdfs = sorted(
    [f for f in os.listdir(RAW) if f.endswith("_magnitude.bdf")],
    key=lambda f: int(re.search(r"p(\d+)", f).group(1)),
)
subjects = [re.search(r"(p\d+)_magnitude", f).group(1) for f in bdfs]
print(f"Found {len(subjects)} subjects: {subjects}")

# ── Per-subject loop ───────────────────────────────────────────────────────
for subj in subjects:
    out_raw    = os.path.join(OUT1,   f"{subj}_raw.fif")
    out_hp     = os.path.join(OUT1,   f"{subj}_hpfilt_raw.fif")
    out_fig    = os.path.join(BRIDGE, f"{subj}_bridging.png")
    out_bridge = os.path.join(BRIDGE, f"{subj}_bridged_pairs.npy")

    if os.path.exists(out_raw):
        print(f"{subj}: already complete — skipping.")
        continue

    print(f"\n{'='*60}\n  Processing {subj}\n{'='*60}")

    # 1. Load raw ────────────────────────────────────────────────────────────
    raw = mne.io.read_raw_bdf(
        os.path.join(RAW, f"{subj}_magnitude.bdf"),
        preload=True, verbose=False,
    )

    # Resample early so all downstream timing is at TARGET_SFREQ
    if subj in RESAMPLE_SUBJECTS:
        print(f"  Resampling {subj} from {raw.info['sfreq']:.0f} → {TARGET_SFREQ} Hz")
        raw.resample(TARGET_SFREQ, npad="auto")

    # 2. Channel setup ───────────────────────────────────────────────────────
    raw.drop_channels(["EXG5", "EXG6", "EXG7", "EXG8"])
    mne.set_bipolar_reference(
        raw,
        anode=["EXG1", "EXG3"], cathode=["EXG2", "EXG4"],
        ch_name=["EOGH", "EOGV"],
        copy=False, drop_refs=False,
    )
    raw.set_channel_types({
        "EOGH": "eog", "EOGV": "eog",
        "EXG1": "misc", "EXG2": "misc", "EXG3": "misc", "EXG4": "misc",
        "Status": "stim",
    })
    dict_biosemi = dict(zip(
        raw.copy().pick("eeg").ch_names,
        mne.channels.make_standard_montage("biosemi64").ch_names,
    ))
    raw.rename_channels(dict_biosemi)
    raw.set_montage("biosemi64")

    # 3. Bridging ────────────────────────────────────────────────────────────
    bridged_idx, ed_matrix = mne.preprocessing.compute_bridged_electrodes(raw)
    np.save(out_bridge, np.array(bridged_idx, dtype=object))
    print(f"  Bridged pairs: {bridged_idx}")

    ed_plot = np.full(ed_matrix.shape[1:], np.nan)
    r, c = np.triu_indices(ed_plot.shape[0], 1)
    for i, j in zip(r, c):
        ed_plot[i, j] = np.nanmedian(ed_matrix[:, i, j])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    fig.suptitle(f"{subj} — Electrical Distance Matrix")
    for ax, vmax in zip(axes, [None, 5]):
        im = ax.imshow(ed_plot, aspect="auto", vmax=vmax)
        fig.colorbar(im, ax=ax, label=r"Elec. Distance ($\mu V^2$)")
        ax.set_xlabel("Channel Index")
        ax.set_ylabel("Channel Index")
    plt.savefig(out_fig, dpi=100)
    plt.close(fig)

    # 4. Event recovery ──────────────────────────────────────────────────────
    events = mne.find_events(raw, verbose=False)
    events = events[events[:, 2] != 50]          # remove offset triggers

    # Build trial map BEFORE any removal: stim_sample → trial number (1-indexed).
    # The nth stim trigger in recording order is behavioral trial n.
    stim_initial = events[np.isin(events[:, 2], STIM_CODES)]
    trial_map    = {str(int(s)): i + 1 for i, s in enumerate(stim_initial[:, 0])}
    removal_log  = {
        "total_initial_trials": len(trial_map),
        "removed": [],
    }
    print(f"  Initial stim events (= behavioural trials): {len(trial_map)}")

    # Orphaned stimuli: stim not immediately followed by a response
    orphan = []
    for i in range(len(events) - 1):
        if events[i, 2] in STIM_CODES and events[i + 1, 2] in STIM_CODES:
            orphan.append(i)
    if events[-1, 2] in STIM_CODES:           
        orphan.append(len(events) - 1)
    print(f"  Orphaned triggers removed: {len(orphan)}")

    # Log before deleting so sample numbers are still valid
    for i in orphan:
        sample = str(int(events[i, 0]))
        removal_log["removed"].append({
            "trial_number": trial_map[sample],
            "reason":       "orphaned_trigger",
            "stim_sample":  int(sample),
        })
    for i in sorted(orphan, reverse=True):
        events = np.delete(events, i, axis=0)

    # Orphaned responses: resp not immediately preceded by a stim
    resp_orphan = []
    for i in range(len(events)):
        if events[i, 2] in RESP_CODES:
            if i == 0 or events[i - 1, 2] not in STIM_CODES:
                resp_orphan.append(i)
    if resp_orphan:
        print(f"  Orphaned responses removed: {len(resp_orphan)}")
        for i in sorted(resp_orphan, reverse=True):
            events = np.delete(events, i, axis=0)

    # 5. RT outlier removal (>3 SD from subject mean) ────────────────────────
    stim_ev = events[np.isin(events[:, 2], STIM_CODES)]
    resp_ev = events[np.isin(events[:, 2], RESP_CODES)]
    rts     = (resp_ev[:, 0] - stim_ev[:, 0]) / raw.info["sfreq"]
    mean_rt, sd_rt = rts.mean(), rts.std()
    outlier_mask   = (rts > mean_rt + 3 * sd_rt) | (rts < mean_rt - 3 * sd_rt)
    print(f"  RT outliers removed: {outlier_mask.sum()}")

    bad_stim_samples = stim_ev[outlier_mask, 0]
    for s in bad_stim_samples:
        sample = str(int(s))
        removal_log["removed"].append({
            "trial_number": trial_map[sample],
            "reason":       "rt_outlier",
            "stim_sample":  int(sample),
        })

    stim_pos = np.where(np.isin(events[:, 0], bad_stim_samples)
                        & np.isin(events[:, 2], STIM_CODES))[0]
    resp_pos = stim_pos + 1
    rm_idx   = np.sort(np.unique(np.concatenate([stim_pos, resp_pos])))[::-1]
    for i in rm_idx:
        events = np.delete(events, i, axis=0)

    raw.add_events(events, replace=True)

    # 6. Annotate inter-trial breaks ─────────────────────────────────────────
    stim_ev = events[np.isin(events[:, 2], STIM_CODES)]
    resp_ev = events[np.isin(events[:, 2], RESP_CODES)]
    stim_t  = stim_ev[:, 0] / raw.info["sfreq"]
    resp_t  = resp_ev[:, 0] / raw.info["sfreq"]

    # keep 400 ms prior to stim onset and 400 ms after response
    onset_dur = np.concatenate([[0],       resp_t + 0.4])
    offset_pt = np.concatenate([stim_t - 0.4, [raw.times.max()]])
    durations = offset_pt - onset_dur

    # Check that breaks are non-negative
    n_neg = (durations < 0).sum()
    if n_neg:
        print(f"  Warning: {n_neg} negative break duration(s) clipped to 0")
    durations = np.clip(durations, 0, None)

    raw.set_annotations(mne.Annotations(
        onset=onset_dur, duration=durations, description="BAD_breaks",
    ))

    # 7. HP-filtered copy for bad-channel inspection (saved BEFORE band-pass) ─
    raw_hp = raw.copy()
    raw_hp.filter(
        0.5, None, l_trans_bandwidth="auto",
        filter_length="auto", phase="zero", n_jobs=8, verbose=False,
    )
    raw_hp.save(out_hp, overwrite=False)
    del raw_hp
    print(f"  Saved HP copy: {out_hp}")

    # 8. Band-pass filter ────────────────────────────────────────────────────
    raw.filter(
        0.5, 40, l_trans_bandwidth="auto",
        filter_length="auto", phase="zero", n_jobs=8, verbose=False,
    )

    # 9. Save raw + trial tracking files ────────────────────────────────────
    # REST re-referencing is applied in stage 3, after bad channels are known.
    raw.save(out_raw, overwrite=False)
    print(f"  Saved: {out_raw}")

    # trial_map.json  — stim_sample (str) → original behavioural trial number
    # removal_log.json — log of every trial removed and why (appended to by
    #                    later stages; noisy epochs are added by the notebook)
    trial_map_path   = os.path.join(MANUAL, f"{subj}_trial_map.json")
    removal_log_path = os.path.join(MANUAL, f"{subj}_removal_log.json")
    with open(trial_map_path,   "w") as f: json.dump(trial_map,   f)
    with open(removal_log_path, "w") as f: json.dump(removal_log, f, indent=2)
    print(f"  Trial map saved ({len(trial_map)} entries): {trial_map_path}")
    n_rm = len(removal_log["removed"])
    print(f"  Removal log saved ({n_rm} removals so far): {removal_log_path}")

print("\n✓ Stage 1 complete.")
