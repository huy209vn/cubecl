//! Benchmarks for the RMSNorm implementation.
//!
//! These benches focus on CPU execution to validate launch overhead and cooperative
//! reduction behavior for moderately sized shapes that mirror common transformer
//! workloads.

use std::mem::size_of;

use bytemuck::cast_slice;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use cubecl_cpu::{CpuDevice, CpuRuntime};
use cubecl_runtime::client::ComputeClient;
use cubecl_std::tensor::{self, TensorHandle};

const EPSILON: f32 = 1e-5;

fn prepare_case(
    client: &ComputeClient<CpuRuntime::Server>,
    shape: &[usize],
    with_bias: bool,
) -> (
    TensorHandle<CpuRuntime, f32>,
    TensorHandle<CpuRuntime, f32>,
    Option<TensorHandle<CpuRuntime, f32>>,
    TensorHandle<CpuRuntime, f32>,
) {
    let total_elems: usize = shape.iter().product();
    let axis = *shape.last().expect("shape must not be empty");

    let input: Vec<f32> = (0..total_elems)
        .map(|idx| {
            let signal = (idx as f32 * 0.137_f32).sin();
            let offset = (idx % axis.max(1)) as f32 * 0.021;
            signal + offset
        })
        .collect();

    let weight: Vec<f32> = (0..axis)
        .map(|idx| 0.75_f32 + (idx as f32 % 11.0) * 0.02)
        .collect();

    let bias_values: Option<Vec<f32>> = if with_bias {
        Some(
            (0..axis)
                .map(|idx| -0.1_f32 + (idx as f32 % 7.0) * 0.015)
                .collect(),
        )
    } else {
        None
    };

    let input_alloc = client.create_tensor(cast_slice(&input), shape, size_of::<f32>());
    let input_handle = TensorHandle::<CpuRuntime, f32>::new(
        input_alloc.handle,
        shape.to_vec(),
        input_alloc.strides,
    );

    let weight_alloc = client.create_tensor(cast_slice(&weight), &[axis], size_of::<f32>());
    let weight_handle =
        TensorHandle::<CpuRuntime, f32>::new(weight_alloc.handle, vec![axis], weight_alloc.strides);

    let bias_handle = bias_values.map(|bias_vec| {
        let bias_alloc = client.create_tensor(cast_slice(&bias_vec), &[axis], size_of::<f32>());
        TensorHandle::<CpuRuntime, f32>::new(bias_alloc.handle, vec![axis], bias_alloc.strides)
    });

    let output_handle = TensorHandle::<CpuRuntime, f32>::empty(client, shape.to_vec());

    (input_handle, weight_handle, bias_handle, output_handle)
}

fn bench_rms_norm(c: &mut Criterion) {
    let device = CpuDevice::default();
    let client = CpuRuntime::client(&device);

    let cases: [(&str, &[usize], bool); 4] = [
        ("mid_no_bias", &[4, 4, 128], false),
        ("mid_bias", &[4, 4, 128], true),
        ("large_no_bias", &[32, 512, 1024], false),
        ("large_bias", &[32, 512, 1024], true),
    ];

    for (label, shape, with_bias) in cases {
        let case_client = client.clone();
        let (input_handle, weight_handle, bias_handle, output_handle) =
            prepare_case(&case_client, shape, with_bias);

        let bench_id = format!("rms_norm_{}", label);

        c.bench_function(&bench_id, move |b| {
            let client = case_client.clone();
            let input = input_handle.clone();
            let weight = weight_handle.clone();
            let bias = bias_handle.clone();
            let output = output_handle.clone();

            b.iter(|| {
                tensor::rms_norm::launch::<CpuRuntime, f32>(
                    &client,
                    &input,
                    &weight,
                    bias.as_ref(),
                    &output,
                    EPSILON,
                );

                black_box(&output);
            });
        });
    }
}

criterion_group!(benches, bench_rms_norm);
criterion_main!(benches);
