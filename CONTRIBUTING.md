# Contributing to AI Cluster

We welcome contributions to AI Cluster! Whether it's fixing bugs, improving documentation, or adding new features, your help is appreciated.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally.
3.  **Set up your environment**:
    *   **Python**: Install dependencies with `pip install -r coordinator/requirements.txt`.
    *   **Rust**: Ensure you have Rust 1.70+ installed.
    *   **Pre-commit**: We use pre-commit hooks for formatting and linting. Install with `pip install pre-commit && pre-commit install`.

## Development Workflow

### Coordinator (Python)
The coordinator is built with FastAPI.
```bash
cd coordinator
uvicorn main:app --reload
```

### Worker (Rust)
The worker is built with Burn.
```bash
cd worker
cargo build
cargo test
```

### Protocol Buffers
If you modify `.proto` files, regenerate the bindings:
```bash
./scripts/generate_protos.sh
```

## Pull Request Process

1.  Create a new branch for your feature or fix.
2.  Write tests for your changes.
3.  Ensure all tests pass.
4.  Submit a Pull Request (PR) with a clear description of changes.

## Running Tests
We have a suite of tests to ensure stability.

### Coordinator Tests (Python)
```bash
cd coordinator
pytest
```

### Worker Tests (Rust)
```bash
cd worker
cargo test
```

### Integration Tests
We have end-to-end integration tests in the `tests/` directory.
```bash
python tests/test_client.py
python tests/cluster_chat.py
```

## Helper Scripts
- `scripts/benchmark.py`: Performance benchmarking tool.
- `scripts/convert_model.py`: Convert HuggingFace models to our format.
- `scripts/setup_cuda.sh` / `setup_rocm.sh`: Environment setup.

## Code Style

*   **Python**: We follow PEP 8. Use `black` and `isort`.
*   **Rust**: Follow standard Rust style. Use `cargo fmt` and `cargo clippy`.

## Reporting Issues

Please use the GitHub Issue Tracker to report bugs or request features. Provide as much detail as possible, including logs and reproduction steps.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

### Contributor License Agreement (CLA)

To accept your contributions, we require you to sign our Contributor License Agreement (CLA). This ensures that we have the necessary rights to use your contribution and can safely distribute the project under the current license.

A bot will prompt you to sign the CLA when you open a Pull Request. We cannot merge any PRs without a signed CLA.
