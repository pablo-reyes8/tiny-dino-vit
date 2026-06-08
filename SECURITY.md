# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| `main` (latest) | Yes |
| Older tags | No — please upgrade |

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public GitHub issue**.

Instead, report it privately by emailing the maintainers at the address listed in the repository contact information. Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- Any suggested mitigation or fix

You will receive an acknowledgement within **72 hours**. We aim to release a patch within **14 days** of a confirmed vulnerability.

## Scope

This project is a research / training framework. The primary attack surfaces are:

- **Model checkpoints** — do not load `.pt` / `.pth` files from untrusted sources. PyTorch `torch.load` uses Python's `pickle` module, which can execute arbitrary code. Use `weights_only=True` where possible.
- **API server** — the FastAPI server (`src/app/`) is intended for internal or controlled deployment. Do not expose it on a public network without authentication and rate limiting.
- **Serialised configs** — YAML configs loaded from disk are parsed with PyYAML. Avoid loading configs from untrusted sources, or use `yaml.safe_load` (already the project default).

## Out of Scope

Vulnerabilities in third-party dependencies (PyTorch, Hugging Face `datasets`, FastAPI) should be reported upstream to those projects.
