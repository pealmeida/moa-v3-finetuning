# Contributing to GateSwarm MoA Router

Thank you for your interest in contributing! This project aims to make AI model routing more efficient through data-driven heuristic optimization.

## How to Contribute

### Reporting Issues

- **Bugs:** Include the handler version, dataset used, and error output
- **Feature requests:** Describe the use case and expected behavior
- **Performance:** Include dataset size, platform, and timing measurements

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-idea`)
3. Make your changes
4. Test locally: `python handler.py` or `python handler_v31.py`
5. Commit with a descriptive message
6. Open a Pull Request

### Code Style

- **Python:** PEP 8, type hints where practical
- **Docstrings:** Include for all public functions
- **Imports:** Standard library → third-party → local

### Testing

Before submitting:

```bash
# Test v3.0 handler (small dataset)
python handler.py

# Test v3.1 handler (GPD subset)
python handler_v31.py  # or: python -c "from handler_v31 import handler; handler({'input':{'datasets':['gpd'],'max_per':1000}})"

# Test LLMFit tools
python llmfit/llmfit.py --help
python llmfit/anonymizer.py --help
python llmfit/datasets/gpd_generator.py --trivial 100 --light 50
```

## Areas for Contribution

### High Priority
- **Multi-dataset training pipeline** — Combine Alpaca + Self-Instruct + GPD for full-tier coverage
- **Online learning** — Use the feedback buffer (`self_eval.py`) for incremental weight updates
- **New dataset connectors** — Add support for more HF datasets or custom formats

### Medium Priority
- **LLM-assisted labeling** — Mode B labeling with an LLM for uncertain samples
- **Sklearn classifier** — Train a RandomForest/XGBoost model as L2 accuracy tier
- **LoRA export** — Generate Alpaca-format datasets for LLM fine-tuning
- **CI/CD** — Automated testing and Docker image building

### Low Priority
- **Web UI** — Visual dashboard for training results
- **Benchmark suite** — Standardized prompt set for comparing versions
- **NVIDIA classifier** — Re-evaluate the DeBERTa complexity scorer

## Dataset Contributions

If you contribute a dataset:

1. **Anonymize it first** — Use `llmfit/anonymizer.py` to strip PII/secrets
2. **Include a generator** — If synthetic, provide the generation script
3. **Document the distribution** — Include label counts and tier breakdown
4. **Keep it small** — Prefer generators over committing large JSONL files

## Architecture

See `docs/ARCHITECTURE_V3_1.md` for the full v3.1 specification.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
