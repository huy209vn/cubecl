//! GPU benchmarks for the CubeCL RMSNorm kernel on CUDA devices.
//!
//! The benches allocate representative transformer workloads directly on the
//! GPU, warm the kernel once to amortize compilation, and then measure fused
//! RMSNorm launches while synchronizing the CUDA stream each iteration.

use std::{
    mem::size_of,
    panic::{self, AssertUnwindSafe},
};

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use cubecl_common::{device::Device, reader::read_sync};
use cubecl_core::{CubeElement, Runtime, prelude::Float};
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cubecl_runtime::client::ComputeClient;
use cubecl_std::tensor::{self, TensorHandle};
use half::{bf16, f16};

const EPSILON: f32 = 1e-5;

fn try_suppress_panic<T>(f: impl FnOnce() -> T) -> Option<T> {
    let previous_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(AssertUnwindSafe(f)).ok();
    panic::set_hook(previous_hook);
    result
}

type CudaClient = ComputeClient<<CudaRuntime as Runtime>::Server>;

struct PreparedCase<F>
where
    F: Float + CubeElement,
{
    input: TensorHandle<CudaRuntime, F>,
    weight: TensorHandle<CudaRuntime, F>,
    bias: Option<TensorHandle<CudaRuntime, F>>,
    output: TensorHandle<CudaRuntime, F>,
}

fn prepare_case<F>(client: &CudaClient, shape: &[usize], with_bias: bool) -> PreparedCase<F>
where
    F: Float + CubeElement,
{
    let total_elems: usize = shape.iter().product();
    let axis = *shape.last().expect("shape must not be empty");

    let input: Vec<F> = (0..total_elems)
        .map(|idx| {
            let signal = (idx as f32 * 0.137_f32).sin();
            let offset = (idx % axis.max(1)) as f32 * 0.021;
            F::new(signal + offset)
        })
        .collect();

    let weight: Vec<F> = (0..axis)
        .map(|idx| F::new(0.75_f32 + (idx as f32 % 11.0) * 0.02))
        .collect();

    let bias_values: Option<Vec<F>> = if with_bias {
        Some(
            (0..axis)
                .map(|idx| F::new(-0.1_f32 + (idx as f32 % 7.0) * 0.015))
                .collect(),
        )
    } else {
        None
    };

    let input_allocation = client.create_tensor(F::as_bytes(&input), shape, size_of::<F>());
    let input_handle = TensorHandle::<CudaRuntime, F>::new(
        input_allocation.handle,
        shape.to_vec(),
        input_allocation.strides,
    );

    let weight_allocation = client.create_tensor(F::as_bytes(&weight), &[axis], size_of::<F>());
    let weight_handle = TensorHandle::<CudaRuntime, F>::new(
        weight_allocation.handle,
        vec![axis],
        weight_allocation.strides,
    );

    let bias_handle = bias_values.as_ref().map(|bias| {
        let allocation = client.create_tensor(F::as_bytes(bias), &[axis], size_of::<F>());
        TensorHandle::<CudaRuntime, F>::new(allocation.handle, vec![axis], allocation.strides)
    });

    let output_handle = TensorHandle::<CudaRuntime, F>::empty(client, shape.to_vec());

    // Ensure all uploads complete before benchmarking.
    read_sync(client.sync());

    PreparedCase {
        input: input_handle,
        weight: weight_handle,
        bias: bias_handle,
        output: output_handle,
    }
}

const CASES: [(&str, [usize; 3], bool); 5] = [
    ("mid_no_bias", [4, 4, 128], false),
    ("mid_bias", [4, 4, 128], true),
    ("large_no_bias", [32, 512, 1024], false),
    ("large_bias", [32, 512, 1024], true),
    ("deep_no_bias", [1, 1, 8192], false),
];

fn bench_dtype<F>(c: &mut Criterion, dtype_label: &str, client: &CudaClient)
where
    F: Float + CubeElement,
{
    for (label, shape, with_bias) in CASES.iter() {
        let case_client = client.clone();
        let PreparedCase {
            input,
            weight,
            bias,
            output,
        } = prepare_case::<F>(&case_client, shape, *with_bias);

        // Warm the kernel once to avoid measuring compilation overhead.
        {
            let warm_client = case_client.clone();
            tensor::rms_norm::launch::<CudaRuntime, F>(
                &warm_client,
                &input,
                &weight,
                bias.as_ref(),
                &output,
                EPSILON,
            );
            read_sync(warm_client.sync());
        }

        let bench_id = format!("cuda_rms_norm_{}_{}", dtype_label, label);
        c.bench_function(&bench_id, move |b| {
            let client = case_client.clone();
            let input = input.clone();
            let weight = weight.clone();
            let bias = bias.clone();
            let output = output.clone();

            b.iter(|| {
                tensor::rms_norm::launch::<CudaRuntime, F>(
                    &client,
                    &input,
                    &weight,
                    bias.as_ref(),
                    &output,
                    EPSILON,
                );
                read_sync(client.sync());
                black_box(&output);
            });
        });
    }
}

fn bench_rms_norm(c: &mut Criterion) {
    let device_count = try_suppress_panic(|| <CudaDevice as Device>::device_count(0)).unwrap_or(0);

    if device_count == 0 {
        eprintln!(
            "Skipping CUDA RMSNorm benchmarks because no CUDA devices were detected or the driver is unavailable."
        );
        return;
    }

    let Some(client) = try_suppress_panic(|| {
        let device = CudaDevice::default();
        CudaRuntime::client(&device)
    }) else {
        eprintln!(
            "Skipping CUDA RMSNorm benchmarks because the CUDA driver could not be initialized."
        );
        return;
    };

    bench_dtype::<f32>(c, "f32", &client);
    bench_dtype::<f16>(c, "f16", &client);
    bench_dtype::<bf16>(c, "bf16", &client);
}

criterion_group!(benches, bench_rms_norm);
criterion_main!(benches);
