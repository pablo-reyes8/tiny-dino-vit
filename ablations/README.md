# Ablations

A config-driven ablation harness for DINO. Each study varies **one axis** while
holding everything else at the base config, runs each variant as an isolated
training process, and aggregates the runs into a side-by-side comparison.

The harness requires **no changes to the training pipeline** — every knob below
is already exposed by the `train_dino()` wrapper through the standard config
sections. An ablation is just a config override plus an isolated checkpoint dir.

---

## The studies

| # | Ablation | Axis (config key) | Variants |
|---|---|---|---|
| 1 | **`gram_loss`** (primary) | `dino.use_gram_loss` / `dino.gram_loss_weight` | baseline, weight_0p05, weight_0p1 |
| 2 | `pos_embed` | `model.pos_embed_type` | learned, rope, none |
| 3 | `register_tokens` | `model.num_register_tokens` | reg_0, reg_4, reg_8 |
| 4 | `local_crops` | `dino.num_local_crops` | local_0, local_2, local_4, local_8 |
| 5 | `teacher_temp` | `dino.teacher_temp` | temp_0p02, temp_0p04, temp_0p07 |
| 6 | `head_out_dim` | `dino_head.out_dim` | dim_1024, dim_4096, dim_8192 |

Each study, its hypothesis, and its expected outcome are declared in
[`ablations.yaml`](ablations.yaml). Edit that file to add variants or new axes —
no Python changes needed.

### Primary hypothesis

The central hypothesis of this project is that **DINOv3-style patch Gram
regularization improves dense feature quality without destabilising training**.
The `gram_loss` study is the one we use to answer it: it compares the standard
DINO objective (`baseline`) against two Gram-loss weights. We read the result
off two signals in the comparison table:

- **`best_loss`** — Gram-regularised runs should reach a lower self-distillation
  loss than the baseline.
- **`cls_std`** (mean per-dimension std of student CLS features) — should be
  *higher* with Gram loss, indicating richer, less-collapsed features.

---

## Running

All commands are run from the project root. Provide short `--epochs` /
`--max-batches-per-epoch` for quick comparisons; drop them for full runs.

```bash
# List every study and its variants
python ablations/run_ablation.py --list

# Run all variants of the primary study (quick)
python ablations/run_ablation.py gram_loss --epochs 30 --max-batches-per-epoch 200

# Run a single variant
python ablations/run_ablation.py gram_loss --variant weight_0p05

# Run every study, then auto-aggregate results
python ablations/run_all.py --epochs 30 --max-batches-per-epoch 200

# Preview the generated configs + commands without training
python ablations/run_ablation.py --all --dry-run
```

Useful flags (`run_ablation.py`):

| Flag | Effect |
|---|---|
| `--all` | Run every study in the registry |
| `--variant NAME` | Run only one variant of the chosen study |
| `--epochs N` | Override epochs for every run |
| `--max-batches-per-epoch N` | Cap batches/epoch for fast smoke runs |
| `--base-config PATH` | Use a different base config (default: `configs/set_up.yaml`) |
| `--device cuda\|cpu` | Force the device |
| `--dry-run` | Generate configs and print commands only |
| `--continue-on-error` | Keep going if one run fails |

---

## How it works

```
ablations.yaml ──► run_ablation.py ──► ablations/_generated/<study>__<variant>.yaml
                                              │
                                              ▼
                              scripts/train_dino.py --config <generated>
                                              │
                                              ▼
                       checkpoints/ablations/<study>/<variant>/metrics.jsonl
                                              │
                       collect_results.py ◄───┘
                                              │
                                              ▼
                       ablations/results/{summary.md, summary.csv, summary.json}
```

1. **`run_ablation.py`** deep-merges each variant's `override` into the base
   config, points `ckpt_dir`/`run_name` at an isolated directory, writes the
   merged config to `ablations/_generated/`, and launches `scripts/train_dino.py`
   in a fresh subprocess (clean CUDA context and seed per run).
2. Training writes per-epoch records to `metrics.jsonl` exactly as for any normal
   run — the harness adds nothing to the training loop.
3. **`collect_results.py`** reads every run's `metrics.jsonl`, reduces it to a
   summary row, and writes a comparison grouped by study, with a Δ column versus
   each study's `reference` variant.

---

## Reading the results

`collect_results.py` prints a Markdown report and writes three files to
`ablations/results/`:

| File | Use |
|---|---|
| `summary.md` | Human-readable comparison, one table per study |
| `summary.csv` | Flat rows for spreadsheets / plotting |
| `summary.json` | Nested structure for programmatic analysis |

Columns:

| Column | Meaning |
|---|---|
| `best_loss` | Lowest monitored loss across epochs (**lower is better**) |
| `Δbest` | Difference vs the study's `reference` variant (✓ better / ✗ worse) |
| `loss_dino` | Final DINO self-distillation loss |
| `loss_gram` | Final Gram loss (0 when disabled) |
| `cls_std` | Mean per-dim std of student CLS features (**higher = less collapse**) |
| `student_H` | Normalised student output entropy |
| `time(s)` | Total wall-clock training time for the run |

```bash
# Aggregate whatever runs currently exist
python ablations/collect_results.py
```

Runs that have not been executed yet show as `no-metrics` rather than failing,
so you can aggregate partial sweeps at any time.

---

## Notes

- Generated configs (`_generated/`) and aggregated results (`results/`) are
  git-ignored; they are reproducible from `ablations.yaml` + the runs.
- Each study includes its own `reference` variant so its table is
  self-contained; the baseline is intentionally repeated across studies.
- To change the base model (e.g. ablate on ViT-Small instead of ViT-Tiny), pass
  `--base-config configs/vit_small.yaml`.
