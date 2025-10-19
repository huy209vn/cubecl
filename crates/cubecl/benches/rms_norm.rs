use std::convert::TryInto;
use std::env;

const DEFAULT_SHAPES: &[[usize; 3]] = &[[4, 4, 128], [8, 1024, 4096], [1, 1, 8192]];
const DEFAULT_WARMUP: usize = 1;

struct BenchConfig {
    shapes: Vec<[usize; 3]>,
    warmup: usize,
    check: bool,
}

fn load_config() -> BenchConfig {
    let shapes = env::var("CUBECL_RMS_SHAPES")
        .ok()
        .map(|raw| parse_shapes(&raw))
        .unwrap_or_else(|| DEFAULT_SHAPES.to_vec());

    let warmup = env::var("CUBECL_RMS_WARMUP")
        .ok()
        .and_then(|raw| match raw.trim().parse::<usize>() {
            Ok(val) => Some(val),
            Err(_) => {
                eprintln!(
                    "Warning: failed to parse CUBECL_RMS_WARMUP='{}', falling back to {DEFAULT_WARMUP}",
                    raw.trim()
                );
                None
            }
        })
        .unwrap_or(DEFAULT_WARMUP);

    let check = matches!(env::var("CUBECL_RMS_CHECK").as_deref(), Ok("1"));

    BenchConfig {
        shapes,
        warmup,
        check,
    }
}

fn parse_shapes(raw: &str) -> Vec<[usize; 3]> {
    let mut shapes = Vec::new();

    for item in raw.split(';') {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parts: Vec<_> = trimmed.split(',').map(|part| part.trim()).collect();
        if parts.len() != 3 {
            eprintln!(
                "Warning: skipping shape '{trimmed}' (expected three comma-separated values)"
            );
            continue;
        }

        let parsed: Option<[usize; 3]> = parts
            .iter()
            .map(|value| value.parse::<usize>().ok())
            .collect::<Option<Vec<_>>>()
            .and_then(|vals| vals.try_into().ok());

        match parsed {
            Some(shape) => shapes.push(shape),
            None => eprintln!("Warning: skipping shape '{trimmed}' (failed to parse usize values)"),
        }
    }

    if shapes.is_empty() {
        DEFAULT_SHAPES.to_vec()
    } else {
        shapes
    }
}

#[cfg(any(
    feature = "wgpu",
    feature = "wgpu-spirv",
    feature = "wgpu-msl",
    feature = "cuda",
    feature = "hip"
))]
mod bench_impl {
    use super::BenchConfig;
    use core::any::{type_name, Any, TypeId};
    use core::marker::PhantomData;
    use core::mem::size_of;
    use cubecl::benchmark::{Benchmark, BenchmarkComputations, TimingMethod};
    use cubecl::frontend;
    use cubecl::future;
    use cubecl::prelude::*;
    use cubecl_random::random_uniform;
    use cubecl_std::tensor::{rms_norm, TensorHandle};
    use half::{bf16, f16};
    use std::panic::{self, AssertUnwindSafe};

    const DEFAULT_EPSILON: f32 = 1e-5;

    struct RmsNormBench<R: Runtime, F: frontend::Float> {
        shape: Vec<usize>,
        with_bias: bool,
        epsilon: f32,
        device: R::Device,
        client: ComputeClient<R::Server>,
        _marker: PhantomData<F>,
    }

    impl<R: Runtime, F: frontend::Float> RmsNormBench<R, F> {
        fn new(device: R::Device, shape: Vec<usize>, with_bias: bool, epsilon: f32) -> Self {
            let client = R::client(&device);
            Self {
                shape,
                with_bias,
                epsilon,
                device,
                client,
                _marker: PhantomData,
            }
        }
    }

    impl<R: Runtime, F: frontend::Float> Benchmark for RmsNormBench<R, F> {
        type Input = (
            TensorHandle<R, F>,
            TensorHandle<R, F>,
            Option<TensorHandle<R, F>>,
            TensorHandle<R, F>,
        );
        type Output = ();

        fn prepare(&self) -> Self::Input {
            let client = R::client(&self.device);
            let input = TensorHandle::<R, F>::empty(&client, self.shape.clone());
            random_uniform::<R, F>(&client, F::from_int(-1), F::from_int(1), input.as_ref());

            let axis = *self
                .shape
                .last()
                .expect("shape must have at least one dimension");
            let weight = TensorHandle::<R, F>::empty(&client, vec![axis]);
            random_uniform::<R, F>(&client, F::from_int(0), F::from_int(1), weight.as_ref());

            let bias = if self.with_bias {
                let bias = TensorHandle::<R, F>::empty(&client, vec![axis]);
                random_uniform::<R, F>(&client, F::from_int(-1), F::from_int(1), bias.as_ref());
                Some(bias)
            } else {
                None
            };

            let output = TensorHandle::<R, F>::empty(&client, self.shape.clone());

            (input, weight, bias, output)
        }

        fn execute(
            &self,
            (input, weight, bias, output): Self::Input,
        ) -> Result<Self::Output, String> {
            let bias_ref = bias.as_ref().map(|b| b.as_ref());
            rms_norm::launch_ref::<R, F>(
                &self.client,
                input.as_ref(),
                weight.as_ref(),
                bias_ref,
                output.as_ref(),
                self.epsilon,
            );
            Ok(())
        }

        fn name(&self) -> String {
            let client = R::client(&self.device);
            format!(
                "{}-rms-norm-{}-bias-{}",
                R::name(&client),
                type_name::<F>(),
                self.with_bias
            )
            .to_lowercase()
        }

        fn options(&self) -> Option<String> {
            Some(format!("bias={}", self.with_bias))
        }

        fn shapes(&self) -> Vec<Vec<usize>> {
            vec![self.shape.clone()]
        }

        fn sync(&self) {
            future::block_on(self.client.sync())
        }
    }

    fn run_bench_for_shape<R, F>(
        device: &R::Device,
        shape: &[usize; 3],
        with_bias: bool,
        warmup: usize,
    ) -> Result<(), String>
    where
        R: Runtime,
        F: frontend::Float + 'static,
    {
        let bench =
            RmsNormBench::<R, F>::new(device.clone(), shape.to_vec(), with_bias, DEFAULT_EPSILON);

        if warmup > 0 {
            let args = bench.prepare();
            for _ in 0..warmup {
                bench.execute(args.clone())?;
            }
            bench.sync();
        }

        match bench.run(TimingMethod::System) {
            Ok(durations) => {
                let computed = BenchmarkComputations::new(&durations);
                let elements: usize = shape.iter().product();
                let bytes_per_elem = size_of::<F>() as f64;
                let traffic_terms = if with_bias { 4.0 } else { 3.0 };
                let traffic_bytes = elements as f64 * bytes_per_elem * traffic_terms;
                let flop_factor = if with_bias { 5.0 } else { 4.0 };
                let flops = elements as f64 * flop_factor;
                let median_secs = computed.median.as_secs_f64();
                let elems_per_sec = if median_secs > 0.0 {
                    elements as f64 / median_secs
                } else {
                    f64::INFINITY
                };
                let gbytes_per_sec = if median_secs > 0.0 {
                    traffic_bytes / median_secs / 1e9
                } else {
                    f64::INFINITY
                };
                let gflops = if median_secs > 0.0 {
                    flops / median_secs / 1e9
                } else {
                    f64::INFINITY
                };

                println!(
                    concat!(
                        "shape={:?} bias={} mean={:.3?} median={:.3?} ",
                        "min={:.3?} max={:.3?} elems/s={:.3e} GB/s={:.3} GFLOP/s={:.3}"
                    ),
                    shape,
                    with_bias,
                    computed.mean,
                    computed.median,
                    computed.min,
                    computed.max,
                    elems_per_sec,
                    gbytes_per_sec,
                    gflops
                );

                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    fn run_rms_norm<R, F>(device: R::Device, config: &BenchConfig)
    where
        R: Runtime,
        F: frontend::Float + 'static,
    {
        let client = R::client(&device);
        let props = client.properties();
        println!(
            "{}<{}>: plane_size_min={}, max_cube_dim.x={}, max_units_per_cube={}",
            R::name(&client),
            type_name::<F>(),
            props.hardware.plane_size_min,
            props.hardware.max_cube_dim.x,
            props.hardware.max_units_per_cube
        );

        if config.check {
            match correctness_check::<R, F>(&device) {
                Ok(()) => println!("Correctness check: OK"),
                Err(err) => {
                    println!("Correctness check failed: {err}. Skipping performance runs.");
                    return;
                }
            }
        }

        for &with_bias in &[false, true] {
            for shape in &config.shapes {
                let device_clone = device.clone();
                let shape_clone = *shape;
                let result = panic::catch_unwind(AssertUnwindSafe(|| {
                    run_bench_for_shape::<R, F>(
                        &device_clone,
                        &shape_clone,
                        with_bias,
                        config.warmup,
                    )
                }));

                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => {
                        println!("Skipping shape {:?} bias {}: {err}", shape_clone, with_bias);
                    }
                    Err(panic) => {
                        println!(
                            "Skipping shape {:?} bias {}: not supported/aligned ({})",
                            shape_clone,
                            with_bias,
                            panic_message(&panic)
                        );
                    }
                }
            }
        }
    }

    fn correctness_check<R, F>(device: &R::Device) -> Result<(), String>
    where
        R: Runtime,
        F: frontend::Float + 'static,
    {
        let bench = RmsNormBench::<R, F>::new(device.clone(), vec![1, 2, 8], true, DEFAULT_EPSILON);
        let (input, weight, bias, output) = bench.prepare();
        bench.execute((input.clone(), weight.clone(), bias.clone(), output.clone()))?;
        bench.sync();

        let client = R::client(device);
        let input_vec = read_tensor_to_f32::<R, F>(&client, &input)?;
        let weight_vec = read_tensor_to_f32::<R, F>(&client, &weight)?;
        let bias_vec = match &bias {
            Some(handle) => Some(read_tensor_to_f32::<R, F>(&client, handle)?),
            None => None,
        };
        let output_vec = read_tensor_to_f32::<R, F>(&client, &output)?;

        let expected = rms_norm_reference(
            &input_vec,
            &weight_vec,
            bias_vec.as_deref(),
            &[1, 2, 8],
            DEFAULT_EPSILON,
        );
        let tolerance = 6e-2_f32;

        for (idx, (&actual, &anticipated)) in output_vec.iter().zip(expected.iter()).enumerate() {
            if (actual - anticipated).abs() > tolerance {
                return Err(format!(
                    "element {} mismatch (expected {:.4}, got {:.4})",
                    idx, anticipated, actual
                ));
            }
        }

        Ok(())
    }

    fn read_tensor_to_f32<R, F>(
        client: &ComputeClient<R::Server>,
        tensor: &TensorHandle<R, F>,
    ) -> Result<Vec<f32>, String>
    where
        R: Runtime,
        F: frontend::Float + 'static,
    {
        let bytes = client.read_tensor(vec![tensor.as_copy_descriptor()]);
        let raw = bytes
            .into_iter()
            .next()
            .ok_or_else(|| "empty tensor read".to_string())?;
        let elem_count: usize = tensor.shape.iter().product();
        if raw.len() != elem_count * size_of::<F>() {
            return Err("read size mismatch".to_string());
        }

        if TypeId::of::<F>() == TypeId::of::<f32>() {
            let mut values = vec![0_f32; elem_count];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr() as *const f32,
                    values.as_mut_ptr(),
                    elem_count,
                );
            }
            Ok(values)
        } else if TypeId::of::<F>() == TypeId::of::<f16>() {
            let mut values = Vec::with_capacity(elem_count);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                values.push(f16::from_bits(bits).to_f32());
            }
            Ok(values)
        } else if TypeId::of::<F>() == TypeId::of::<bf16>() {
            let mut values = Vec::with_capacity(elem_count);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                values.push(bf16::from_bits(bits).to_f32());
            }
            Ok(values)
        } else {
            Err(format!(
                "Unsupported dtype for correctness check: {}",
                type_name::<F>()
            ))
        }
    }

    fn rms_norm_reference(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        shape: &[usize; 3],
        epsilon: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; input.len()];
        let last_dim = shape[2];
        let rows = input.len() / last_dim;

        for row in 0..rows {
            let base = row * last_dim;
            let slice = &input[base..base + last_dim];
            let mut sum_sq = 0.0_f32;
            for &value in slice {
                sum_sq += value * value;
            }
            let mean = sum_sq / last_dim as f32;
            let inv_rms = 1.0_f32 / (mean + epsilon).sqrt();

            for idx in 0..last_dim {
                let scaled = slice[idx] * weight[idx] * inv_rms;
                output[base + idx] = if let Some(bias_vals) = bias {
                    scaled + bias_vals[idx]
                } else {
                    scaled
                };
            }
        }

        output
    }

    fn run_or_skip(label: &str, runner: impl FnOnce()) {
        let hook = panic::take_hook();
        panic::set_hook(Box::new(|_| {}));
        let result = panic::catch_unwind(AssertUnwindSafe(runner));
        panic::set_hook(hook);
        if let Err(panic) = result {
            println!("Skipping {label} benchmark: {}", panic_message(&panic));
        }
    }

    fn panic_message(panic: &Box<dyn Any + Send>) -> String {
        if let Some(msg) = panic.downcast_ref::<&str>() {
            (*msg).to_string()
        } else if let Some(msg) = panic.downcast_ref::<String>() {
            msg.clone()
        } else {
            "unknown panic".to_string()
        }
    }

    pub fn run_with_config(config: &BenchConfig) {
        let mut ran = false;

        #[cfg(all(
            feature = "wgpu",
            not(feature = "wgpu-spirv"),
            not(feature = "wgpu-msl")
        ))]
        {
            ran = true;
            run_or_skip("wgpu<f32>", || {
                run_rms_norm::<cubecl::wgpu::WgpuRuntime, f32>(Default::default(), config);
            });
        }

        #[cfg(feature = "wgpu-spirv")]
        {
            ran = true;
            run_or_skip("wgpu-spirv<f16>", || {
                run_rms_norm::<cubecl::wgpu::WgpuRuntime, half::f16>(Default::default(), config);
            });
        }

        #[cfg(feature = "wgpu-msl")]
        {
            ran = true;
            run_or_skip("wgpu-msl<f32>", || {
                run_rms_norm::<cubecl::wgpu::WgpuRuntime, f32>(Default::default(), config);
            });
        }

        #[cfg(feature = "cuda")]
        {
            ran = true;
            run_or_skip("cuda<f32>", || {
                run_rms_norm::<cubecl::cuda::CudaRuntime, f32>(Default::default(), config);
            });
            run_or_skip("cuda<f16>", || {
                run_rms_norm::<cubecl::cuda::CudaRuntime, f16>(Default::default(), config);
            });
            run_or_skip("cuda<bf16>", || {
                run_rms_norm::<cubecl::cuda::CudaRuntime, bf16>(Default::default(), config);
            });
        }

        #[cfg(all(feature = "hip", target_os = "linux"))]
        {
            ran = true;
            run_or_skip("hip<f16>", || {
                run_rms_norm::<cubecl::hip::HipRuntime, half::f16>(Default::default(), config);
            });
        }

        if !ran {
            println!(
                "No CubeCL runtimes enabled; skipping RMSNorm benchmarks for {} shapes.",
                config.shapes.len()
            );
        }
    }
}

#[cfg(not(any(
    feature = "wgpu",
    feature = "wgpu-spirv",
    feature = "wgpu-msl",
    feature = "cuda",
    feature = "hip"
)))]
mod bench_impl {
    use super::BenchConfig;

    pub fn run_with_config(config: &BenchConfig) {
        let _ = config.warmup;
        let _ = config.check;
        println!(
            "No CubeCL runtimes enabled; skipping RMSNorm benchmarks for {} shapes.",
            config.shapes.len()
        );
    }
}

pub fn run() {
    let config = load_config();
    bench_impl::run_with_config(&config);
}

fn main() {
    run();
}
