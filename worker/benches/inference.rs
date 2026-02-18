//! Inference benchmarks.
//!
//! Run with: `cargo bench --bench inference`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn inference_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder_forward_pass", |b| {
        b.iter(|| {
            // Placeholder: actual model forward pass benchmarks will be
            // added once weight loading is implemented.
            let _result = black_box(42);
        });
    });
}

criterion_group!(benches, inference_benchmark);
criterion_main!(benches);
