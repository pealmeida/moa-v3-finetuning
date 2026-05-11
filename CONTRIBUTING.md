# Contributing to GateSwarm MoA Router

Thank you for your interest! This project makes AI model routing more efficient through data-driven complexity classification.

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Test locally:
   ```bash
   # Quick sanity check
   python router.py "test prompt" --json

   # Full test suite
   python test_router.py
   ```
4. Run training validation (if modifying training): `python train.py`
5. Submit a pull request

## Development Setup

```bash
# Runtime only
pip install numpy

# Full development (training + testing)
pip install scipy numpy scikit-learn datasets
python test_router.py
```

## Project Structure

| File | Purpose | Edit with care |
|------|---------|---------------|
| `router.py` | Production scorer — must remain standalone | ⚠️ Zero cloud deps |
| `train.py` | Training pipeline (cascade + optimization) | ✅ |
| `v32_cascade_weights.json` | Pre-trained weights (5 classifiers) | ⚠️ Regenerate via `train.py` |
| `llmfit/` | Dataset factory toolkit | ✅ |
| `test_router.py` | Test suite | ✅ |
| `docs/` | Architecture & strategy docs | ✅ |

## Code Style

- Python 3.11+ with type hints
- Keep `router.py` standalone (zero cloud dependencies)
- Training tools go in `train.py` or `llmfit/`
- New features should include test cases in `test_router.py`

## Reporting Issues

- Bug reports: [GitHub Issues](https://github.com/pealmeida/gateswarm-moa-router/issues)
- Feature requests: GitHub Issues with `enhancement` label
- Security vulnerabilities: See [SECURITY.md](SECURITY.md) — **do not** open public issues

## Pull Request Checklist

- [ ] `python test_router.py` passes
- [ ] No new external dependencies in `router.py`
- [ ] Docstrings updated for new public APIs
- [ ] CHANGELOG.md updated with changes

## License

By contributing, you agree your code is licensed under MIT.
