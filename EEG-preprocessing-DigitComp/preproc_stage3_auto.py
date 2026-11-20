"""
Stage 3 — Automated: apply manual decisions and save final epochs.

Requires per-subject:
  preprocessing/manual/{subj}_bad_info.json      (Phase A)
  preprocessing/manual/{subj}_bad_epochs.json    (Phase B)
  preprocessing/manual/{subj}_ica_exclude.json   (Phase B)
  preprocessing/stage2/{subj}_ica.fif

Produces:
  epoched_data/{subj}-epo.fif

Pipeline order: apply bad channels → remove bad epochs → REST reference →
apply ICA → interpolate bads/bridged → epoch.
REST is applied before ICA (matching stage 2) so ICA is trained and
applied on the same referenced data.

Subjects with missing inputs are skipped so the script can be run
incrementally as Phase B inspection is completed.

Run:
    python stage3_auto.py
"""

import os
import re
import json

import numpy as np
import mne

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = "" # add project path here
OUT1    = os.path.join(BASE, "preprocessing", "stage1")
OUT2    = os.path.join(BASE, "preprocessing", "stage2")
MANUAL  = os.path.join(BASE, "preprocessing", "manual")
EPO_OUT = os.path.join(BASE, "epoched_data")
os.makedirs(EPO_OUT, exist_ok=True)

STIM_CODES = [51, 53]
RESP_CODES = [55, 56]
STIM_ID    = {"stimulus/smallD": 51, "stimulus/largeD": 53}
RESP_ID    = {"response/correct": 56, "response/incorrect": 55}
EVENT_ID   = STIM_ID | RESP_ID

# ── Subject files ──────────────────────────────────────────────────────
raw_files = sorted(
    [f for f in os.listdir(OUT1) if f.endswith("_raw.fif") and "hpfilt" not in f],
    key=lambda f: int(re.search(r"p(\d+)", f).group(1)),
)
subjects = [re.search(r"(p\d+)_raw", f).group(1) for f in raw_files]
print(f"Found {len(subjects)} stage-1 outputs: {subjects}")

# ── Per-subject loop ───────────────────────────────────────────────────────
for subj in subjects:
    out_epo = os.path.join(EPO_OUT, f"{subj}-epo.fif")

    if os.path.exists(out_epo):
        print(f"{subj}: final epochs already exist — skipping.")
        continue

    # Check all required inputs are present
    bad_info_path    = os.path.join(MANUAL, f"{subj}_bad_info.json")
    bad_epochs_path  = os.path.join(MANUAL, f"{subj}_bad_epochs.json")
    ica_exclude_path = os.path.join(MANUAL, f"{subj}_ica_exclude.json")
    ica_path         = os.path.join(OUT2,   f"{subj}_ica.fif")
    trial_map_path   = os.path.join(MANUAL, f"{subj}_trial_map.json")
    removal_log_path = os.path.join(MANUAL, f"{subj}_removal_log.json")

    missing = [p for p in [bad_info_path, bad_epochs_path, ica_exclude_path,
                            ica_path, trial_map_path] if not os.path.exists(p)]
    if missing:
        print(f"{subj}: missing inputs — skipping.\n  {missing}")
        continue

    print(f"\n{'='*60}\n  Processing {subj}\n{'='*60}")

    # Load inputs ────────────────────────────────────────────────────────────
    with open(bad_info_path)    as f: bad_info    = json.load(f)
    with open(bad_epochs_path)  as f: bad_epochs  = json.load(f)
    with open(ica_exclude_path) as f: ica_exclude = json.load(f)
    # trial_map keys are strings (JSON limitation); cast back to int for lookup
    with open(trial_map_path)   as f: trial_map   = {int(k): v for k, v in json.load(f).items()}

    bad_channels     = bad_info["bad_channels"]
    bridged_pairs    = bad_info.get("bridged_pairs", [])
    bad_stim_samples = np.array(bad_epochs["bad_stim_samples"])

    ica = mne.preprocessing.read_ica(ica_path, verbose=False)
    ica.exclude = ica_exclude["ica_exclude"]
    print(f"  Bad channels: {bad_channels}")
    print(f"  Bad epochs:   {len(bad_stim_samples)}")
    print(f"  ICA exclude:  {ica.exclude}")

    # Load stage-1 raw and apply bad channel list ────────────────────────────
    raw = mne.io.read_raw_fif(
        os.path.join(OUT1, f"{subj}_raw.fif"), preload=True, verbose=False,
    )
    raw.info["bads"] = bad_channels

    # Remove bad epochs from the stim channel ────────────────────────────────
    # Bad epochs are stored as stimulus event sample-numbers (timestamps),
    # which are unambiguous regardless of array indexing.
    events = mne.find_events(raw, verbose=False)
    stim_pos = np.where(
        np.isin(events[:, 0], bad_stim_samples) & np.isin(events[:, 2], STIM_CODES)
    )[0]
    # Find the actual response event following each bad stim rather than
    # assuming it is always at stim_pos + 1.
    resp_pos = []
    for sp in stim_pos:
        for j in range(sp + 1, len(events)):
            if events[j, 2] in RESP_CODES:
                resp_pos.append(j)
                break
    rm_idx = np.sort(np.unique(np.concatenate([stim_pos, resp_pos])))[::-1]
    for i in rm_idx:
        events = np.delete(events, i, axis=0)
    raw.add_events(events, replace=True)
    print(f"  Events after bad-epoch removal: {len(events)} ({len(events)//2} trials)")

    # REST reference — applied before ICA to match the reference used during
    # ICA fitting in stage 2.
    sphere = mne.make_sphere_model("auto", "auto", raw.info, verbose=False)
    src    = mne.setup_volume_source_space(sphere=sphere, exclude=30.0,
                                           pos=5.0, verbose=False)
    fwd    = mne.make_forward_solution(raw.info, trans=None, src=src,
                                       bem=sphere, verbose=False)
    raw.set_eeg_reference("REST", forward=fwd, verbose=False)
    print("  REST reference applied.")

    # Apply ICA ──────────────────────────────────────────────────────────────
    ica.apply(raw, verbose=False)

    # Interpolate bad channels and bridged pairs 
    # bridged_pairs comes from bad_info.json (Phase A, manual_inspection)
    raw.interpolate_bads(reset_bads=True, verbose=False)
    if bridged_pairs:
        raw = mne.preprocessing.interpolate_bridged_electrodes(
            raw.copy(), bridged_idx=bridged_pairs,
        )

    # Final epoching with metadata ────────────────────────────────────────────
    events = mne.find_events(raw, verbose=False)
    stim_ev = events[np.isin(events[:, 2], STIM_CODES)]
    resp_ev = events[np.isin(events[:, 2], RESP_CODES)]
    rts     = (resp_ev[:, 0] - stim_ev[:, 0]) / raw.info["sfreq"]
    tmin    = -0.25
    tmax    = rts.max() + 0.1

    metadata, events_m, event_id_m = mne.epochs.make_metadata(
        events=events, event_id=EVENT_ID,
        tmin=tmin, tmax=tmax, sfreq=raw.info["sfreq"],
        row_events=list(STIM_ID.keys()),
        keep_first=["stimulus", "response"],
    )
    epochs = mne.Epochs(
        raw, events_m, event_id_m, tmin, tmax,
        proj=False, baseline=None, preload=True, verbose=False,
        metadata=metadata, reject_by_annotation=False, reject=None,
    )

    # Stamp each epoch with its original behavioural trial number.
    epochs.metadata["trial_number"] = [
        trial_map[int(s)] for s in epochs.events[:, 0]
    ]

    epochs.save(out_epo, overwrite=False)
    print(f"  Saved final epochs ({len(epochs)} trials): {out_epo}")

    # Write a summary of every removed trial
    if os.path.exists(removal_log_path):
        with open(removal_log_path) as f:
            removal_log = json.load(f)
        n_init    = removal_log["total_initial_trials"]
        n_removed = len(removal_log["removed"])
        surviving = sorted(epochs.metadata["trial_number"].tolist())
        print(f"  Trial summary: {n_init} initial → "
              f"{n_removed} removed → {len(epochs)} final")
        removal_log["surviving_trial_numbers"] = surviving
        with open(removal_log_path, "w") as f:
            json.dump(removal_log, f, indent=2)
        print(f"  Removal log finalised: {removal_log_path}")

print("\n✓ Stage 3 complete.")
