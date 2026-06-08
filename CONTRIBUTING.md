# Contributing

Thank you for considering a contribution to this project. All kinds of help are welcome: bug reports, documentation improvements, new configs, and code changes.

## Reporting bugs

Open a GitHub issue and include:

- A minimal reproducible example (config + script invocation that triggers the bug)
- Python / PyTorch / CUDA versions (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`)
- The full traceback

## Proposing features or configs

Open an issue describing the motivation before opening a pull request. For new training configs, include benchmark numbers or at least expected training dynamics.

## Pull requests

### Setup

```bash
git clone https://github.com/your-org/dino.git
cd dino
pip install -e ".[dev]"
```

### Workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes. Keep commits focused and atomic.
3. Run the test suite and linting before opening a PR.
4. Open a pull request against `main` with a clear description of what changed and why.

### Code style

This project uses [Black](https://github.com/psf/black) for formatting and [isort](https://pycqa.github.io/isort/) for import ordering.

```bash
black src/ tests/ scripts/ data/
isort src/ tests/ scripts/ data/
```

### Tests

All new behaviour must be covered by tests under `tests/`. Run the suite with:

```bash
pytest tests/ -v
```

Tests must pass before a PR is merged.

### Commit messages

Use the imperative mood in the subject line (`Add RoPE support`, not `Added RoPE support`). Keep the subject under 72 characters. Reference issue numbers where relevant.

## Code of Conduct

Be respectful and constructive in all interactions. Harassment of any kind will not be tolerated.
