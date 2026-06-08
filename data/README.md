# DataOps

This folder contains dataset loading plus governance artifacts for DINO pretraining.

Standard workflow:

1. Inspect data with `python scripts/inspect_data.py --write-report`.
2. Review `data/reports/data_card.json` before training.
3. Keep generated reports and manifest sidecars with every experiment checkpoint.
4. Use fixed seeds and config JSON files for reproducible train/validation splits.

Governance checks covered here:

- Dataset config validation before ingestion.
- Split/profile summaries without materializing the whole dataset.
- Batch-shape inspection for DINO multi-crop invariants.
- Dirty-image checks: blur, low contrast, under/over-exposure, saturation, tiny images and effectively grayscale images.
- Environment fingerprinting for reproducibility.
- Artifact documentation with size and SHA-256 for files.
- Machine-readable data cards in JSON.

Expected generated artifacts:

- `data/reports/data_profile.json`: split and sampled image/label profile.
- `data/reports/data_card.json`: governance, config, validation and environment metadata.
- `data/reports/batch_inspection.json`: train/eval dataloader batch diagnostics.

Quality flags in `data_profile.json`:

- `blurry`: low Laplacian variance.
- `too_dark`: mean pixel intensity is below threshold.
- `too_bright`: mean pixel intensity is above threshold.
- `low_contrast`: pixel standard deviation is below threshold.
- `saturated`: too many near-black or near-white pixels.
- `too_small`: one image side is below the configured minimum.
- `effectively_grayscale`: RGB channels are effectively identical.

Do not commit raw datasets unless the license and repository policy explicitly allow it.
