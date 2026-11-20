"""
Stage 2 — Automated: temp epochs + ICA fitting

Requires per-subject:  preprocessing/manual/{subj}_bad_info.json
                       (produced by manual_inspection.ipynb, Phase A)

Produces per-subject:
  preprocessing/stage2/{subj}_temp-epo.fif   full-RT epochs for visual inspection
  preprocessing/stage2/{subj}_ica.fif        fitted ICA object

Subjects whose manual JSON is missing are skipped with a warning so the
script can be run incrementally as Phase A is completed.

Run:
    python stage2_auto.py
"""

import os
import re
import json

import numpy as np
import mne

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = "" # add project path here
OUT1   = os.path.join(BASE, "preprocessing", "stage1")
OUT2   = os.path.join(BASE, "preprocessing", "stage2")
MANUAL = os.path.join(BASE, "preprocessing", "manual")

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
    out_epo = os.path.join(OUT2, f"{subj}_temp-epo.fif")
    out_ica = os.path.join(OUT2, f"{subj}_ica.fif")
    bad_info_path = os.path.join(MANUAL, f"{subj}_bad_info.json")

    if os.path.exists(out_epo) and os.path.exists(out_ica):
        print(f"{subj}: already complete — skipping.")
        continue

    if not os.path.exists(bad_info_path):
        print(f"{subj}: bad_info.json not found — run Phase A inspection first.")
        continue

    print(f"\n{'='*60}\n  Processing {subj}\n{'='*60}")

    # Load stage-1 raw and apply bad channels ────────────────────────────────
    raw = mne.io.read_raw_fif(
        os.path.join(OUT1, f"{subj}_raw.fif"), preload=True, verbose=False,
    )
    with open(bad_info_path) as f:
        bad_info = json.load(f)
    raw.info["bads"] = bad_info["bad_channels"]
    print(f"  Bad channels: {raw.info['bads']}")

    # REST reference — applied before ICA so components are learned on
    # referenced data. Bad channels are already marked and excluded from
    # the forward model.
    sphere = mne.make_sphere_model("auto", "auto", raw.info, verbose=False)
    src    = mne.setup_volume_source_space(sphere=sphere, exclude=30.0,
                                           pos=5.0, verbose=False)
    fwd    = mne.make_forward_solution(raw.info, trans=None, src=src,
                                       bem=sphere, verbose=False)
    raw.set_eeg_reference("REST", forward=fwd, verbose=False)
    print("  REST reference applied.")

    # Recover events ─────────────────────────────────────────────────────────
    events = mne.find_events(raw, verbose=False)
    stim_ev = events[np.isin(events[:, 2], STIM_CODES)]
    resp_ev = events[np.isin(events[:, 2], RESP_CODES)]
    rts     = (resp_ev[:, 0] - stim_ev[:, 0]) / raw.info["sfreq"]
    tmin    = -0.25
    tmax    = rts.max() + 0.1

    # Build metadata-aware epoch object ─────────────────────────────────────
    metadata, events_m, event_id_m = mne.epochs.make_metadata(
        events=events, event_id=EVENT_ID,
        tmin=tmin, tmax=tmax, sfreq=raw.info["sfreq"],
        row_events=list(STIM_ID.keys()),
        keep_first=["stimulus", "response"],
    )
    epoch_temp = mne.Epochs(
        raw, events_m, event_id_m, tmin, tmax,
        proj=False, baseline=None, preload=True, verbose=False,
        metadata=metadata, reject_by_annotation=False, reject=None,
    )
    epoch_temp.save(out_epo, overwrite=False)
    print(f"  Saved temp epochs ({len(epoch_temp)} trials): {out_epo}")

    # Fit ICA on a copy with bad channels excluded ───────────────────────────
    raw_ica = raw.copy().pick("eeg", exclude="bads")
    ica = mne.preprocessing.ICA(
        n_components=0.999999, max_iter="auto", random_state=97,
    )
    ica.fit(raw_ica, verbose=False)
    del raw_ica

    ica.save(out_ica, overwrite=False)
    print(f"  Saved ICA ({ica.n_components_} components): {out_ica}")

print("\n✓ Stage 2 complete.")
