# Temporary perception-validation: Task 8 handoff and removal record

This document is the implementation-time record for the temporary validation
workflow. It is not a claim that the workflow has been tested, run against the
supplied media, or scientifically verified. The corresponding evidence file
records that status explicitly.

## Reproducibility/report metadata contract

Every exported or retained validation report should carry one metadata object
with the following shape. Values that were not observed must remain `null`, an
empty collection, or an explicit unavailable status; they must not be inferred
from the plan or filled from the comparison CSV.

```json
{
  "schema_version": "task-8-v1",
  "verification_status": "implementation_only_unverified",
  "session": {
    "session_id": "<uuid-hex>",
    "created_at": "<UTC ISO-8601>",
    "label": "temporary validation",
    "artifact_root": "data/perception-validation/<session_id>"
  },
  "source_video": {
    "original_filename": "<name>",
    "width_px": null,
    "height_px": null,
    "fps": null,
    "frame_count": null,
    "duration_s": null,
    "source_sha256": null
  },
  "selection": {
    "source_start_s": null,
    "source_end_s": null,
    "frame_stride": null,
    "target_fps": null,
    "representative_frame_index": null,
    "representative_timestamp_s": null,
    "sampled_frames": [],
    "timestamp_policy": "preserve_source_frame_timestamps"
  },
  "calibration": {
    "pivot": null,
    "reference_start": null,
    "reference_end": null,
    "endpoint_separation_px": null,
    "confirmed_physical_distance_m": null,
    "metres_per_pixel": null,
    "calibration_type": "unset"
  },
  "leakage_masks": {
    "regions": [],
    "simulator_labels_excluded": false,
    "displayed_parameters_excluded": false,
    "live_angle_values_excluded": false,
    "controls_excluded": false,
    "ground_truth_source": "independent_evaluation_only"
  },
  "sam": {
    "method": null,
    "propagation_direction": null,
    "fallback_label": null,
    "object_id": null,
    "records": null,
    "masks_returned": null,
    "frames_with_masks": null,
    "frames_tracked": null,
    "selected_frame_indices": [],
    "selected_timestamps_s": [],
    "missing_frame_indices": [],
    "rejection_reasons": [],
    "confidence": null,
    "parser_diagnostics": null,
    "stream_event_excerpt": null
  },
  "quality": {
    "status": "not_checked",
    "coverage_fraction": null,
    "radius_stability": "not_checked",
    "discontinuity_count": null,
    "metric_distance_rmse_m": null,
    "length_error_m": null,
    "angular_accuracy_deg": null,
    "reasons": [],
    "sendable": false
  },
  "csv_comparison": {
    "independent_only": true,
    "original_filename": null,
    "timestamp_column": null,
    "angular_column": null,
    "angular_unit": null,
    "row_count": null,
    "alignment_method": null,
    "estimated_time_offset_s": null,
    "alignment_uncertainty_s": null,
    "overlap_duration_s": null,
    "overlap_frame_count": null,
    "angular_rmse_deg": null,
    "angular_max_error_deg": null,
    "mismatch_diagnostics": []
  },
  "production_handoff": {
    "status": "not_requested",
    "production_invoked": false,
    "reason": null,
    "comparison_csv_excluded": true,
    "fit": null,
    "selector": null
  },
  "muse": {
    "status": "not_invoked",
    "model": null,
    "finish_reason": "not_called",
    "request_status": "not_invoked",
    "completion_status": "not_started",
    "response_text_length": 0,
    "error_detail": null,
    "interpretation_only": true
  }
}
```

The selected-frame records must retain each source frame index and original
timestamp. The pixel-space and calibrated branches must reference the same
mask-derived observations. When calibration is unset, metric fields remain
null and pixel-coordinate angles/quality are still allowed. Metric/length
accuracy and angular accuracy remain separate fields. The CSV comparison is
evaluation-only and must not appear in the production handoff measurements.

## Complete removal checklist

Run this checklist only when the temporary workflow is retired. The checklist
is a file inventory, not an instruction to remove user media or normal
ProbeLab data. Delete only generated session artifacts under the isolated
temporary root; do not delete existing files elsewhere in `data/`.

### API routes and application wiring

- [ ] Remove the `perception_validation_router` import and
      `app.include_router(perception_validation_router)` from `probelab/main.py`.
- [ ] Remove the temporary API router in
      `probelab/perception_validation/router.py`, including:
      `POST/GET /api/perception-validation/sessions`,
      `POST /api/perception-validation/sessions/{session_id}/uploads`,
      `POST /annotations`, `POST /geometry-quality`, `POST /handoff`,
      `POST /csv-comparison`, and both representative-frame aliases
      (`POST /representative-frame` and `POST /prepare`).
- [ ] Leave all normal `/api/investigations/*`, `/api/sam-test/*`, static
      serving, and production science routes unchanged.

### Temporary models, services, and adapters

- [ ] Remove the package `probelab/perception_validation/` and its modules:
      `__init__.py`, `models.py`, `router.py`, `storage.py`, `workflow.py`,
      `sam_tracking.py`, `geometry.py`, `comparison.py`, and `service.py`.
- [ ] Remove only imports/references that exist solely for this package; do
      not remove or rewrite `probelab/perception.py`, `science.py`,
      `sam_client.py`, or `meta_client.py` production algorithms.

### Temporary UI, CSS, and JavaScript

- [ ] Remove the `#perception-validation` section from `web/index.html`.
- [ ] Remove the temporary stylesheet/script links from `web/index.html`.
- [ ] Delete `web/perception-validation.css` and
      `web/perception-validation.js`.
- [ ] Remove only the associated temporary panel markup and handlers; retain
      the normal ProbeLab setup, observation, report, and history UI.

### Temporary storage and generated artifacts

- [ ] Remove generated session directories below
      `data/perception-validation/` (session JSON, uploads, temporary files,
      previews, diagnostics, and exports) if they exist.
- [ ] Remove generated `data/engine-test/` artifacts if they belong to this
      temporary validation effort.
- [ ] Do **not** delete or rewrite existing user media, investigation JSON,
      logs, or other files directly under `data/`.
- [ ] Keep the ignore rules for `data/perception-validation/`,
      `data/engine-test/`, pending uploads, SAM test media, and temporary
      `.tmp` files so future generated artifacts are not committed.

### Evidence, reports, and tests

- [ ] Decide whether to archive or remove the Task 1–8 evidence records under
      `.omo/evidence/`; if removed, remove only the temporary validation
      evidence, not unrelated project evidence.
- [ ] Remove the Task 8 ledger entry from `.omo/start-work/ledger.jsonl` only
      if the project process record is being retired with the feature.
- [ ] Remove `tests/test_perception_validation.py` and any other tests whose
      sole purpose is the temporary validation package.
- [ ] Remove this checklist and the Task 8 metadata contract, or archive them
      as the retirement record, according to the project evidence-retention
      decision.

No removal step authorizes deletion of credentials, supplied source media, or
existing user data.
