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
The coordinator is built with FastAPI. Run it from the REPO ROOT (running
`uvicorn main:app` from inside `coordinator/` breaks package imports):
```bash
uvicorn coordinator.main:app --reload
```

### Worker (Rust)
The worker is built with Burn.
```bash
cd worker
cargo build
cargo test
```

### Protocol Buffers
If you modify `proto/cluster.proto`, regenerate BOTH bindings:
```bash
# Python (run from repo root, then keep the package-qualified import):
python -m grpc_tools.protoc -I./proto \
  --python_out=./coordinator/proto --grpc_python_out=./coordinator/proto \
  ./proto/cluster.proto
sed -i 's/^import cluster_pb2 as cluster__pb2$/import coordinator.proto.cluster_pb2 as cluster__pb2/' \
  coordinator/proto/cluster_pb2_grpc.py
# Rust: automatic via worker/build.rs on the next cargo build
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
python tests/interactive_chat.py
```

## Helper Scripts
- `scripts/benchmark.py`: Performance benchmarking tool.
- `scripts/convert_model.py`: Convert HuggingFace models to our format.
- `scripts/setup_cuda.sh` / `setup_rocm.sh`: Environment setup.

## Code Style

*   **Python**: `black --line-length 100`, `ruff` (imports covered by ruff's I-rules), `mypy` strict.
*   **Rust**: Follow standard Rust style. Use `cargo fmt` and `cargo clippy`.

## Reporting Issues

Please use the GitHub Issue Tracker to report bugs or request features. Provide as much detail as possible, including logs and reproduction steps.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
