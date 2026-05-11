# Contributing to GateSwarm MoA Router

Thank you for your interest! This project makes AI model routing more efficient through data-driven complexity classification.

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Test locally: `python router.py "test prompt"`
4. Run training validation: `python train.py`
5. Submit a pull request

## Development Setup

```bash
pip install scipy numpy scikit-learn datasets
python router.py "test prompt" --json
```

## Code Style

- Python 3.11+ with type hints
- Keep `router.py` standalone (zero cloud dependencies)
- Training tools go in `train.py` or `llmfit/`

## Reporting Issues

- Bug reports: GitHub Issues
- Feature requests: GitHub Issues with `enhancement` label

## License

By contributing, you agree your code is licensed under MIT.
