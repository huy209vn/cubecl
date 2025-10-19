use core::any::type_name;
use core::marker::PhantomData;

use cubecl::benchmark::{Benchmark, BenchmarkComputations, TimingMethod};
use cubecl::frontend;
use cubecl::future;
use cubecl::prelude::*;
use cubecl_random::random_uniform;
use cubecl_std::tensor::{TensorHandle, rms_norm};
use std::any::Any;
use std::panic::{self, AssertUnwindSafe};

const SHAPES: &[[usize; 3]] = &[[4, 4, 128], [8, 1024, 4096], [1, 1, 8192]];

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

    fn execute(&self, (input, weight, bias, output): Self::Input) -> Result<Self::Output, String> {
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

fn run_bench_for_shape<R, F>(device: R::Device, shape: &[usize; 3], with_bias: bool)
where
    R: Runtime,
    F: frontend::Float,
{
    let bench = RmsNormBench::<R, F>::new(device.clone(), shape.to_vec(), with_bias, 1e-5);
    println!("shape={shape:?} bias={with_bias}");
    match bench.run(TimingMethod::System) {
        Ok(durations) => {
            let computed = BenchmarkComputations::new(&durations);
            let elements: usize = shape.iter().product();
            let median_secs = computed.median.as_secs_f64();
            let rate = if median_secs > 0.0 {
                elements as f64 / median_secs
            } else {
                f64::INFINITY
            };
            println!(
                "{} -> mean: {:.3?}, median: {:.3?}, min: {:.3?}, max: {:.3?}",
                bench.name(),
                computed.mean,
                computed.median,
                computed.min,
                computed.max
            );
            let durations_secs: Vec<f64> = durations
                .durations
                .iter()
                .map(|d| d.as_secs_f64())
                .collect();
            println!("durations (s): {:?}", durations_secs);
            println!("elements/sec: {:.3e}", rate);
        }
        Err(err) => {
            eprintln!("{}", err);
        }
    }
}

fn run_rms_norm<R, F>(device: R::Device)
where
    R: Runtime,
    F: frontend::Float,
{
    for &with_bias in &[false, true] {
        for shape in SHAPES {
            run_bench_for_shape::<R, F>(device.clone(), shape, with_bias);
        }
    }
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

pub fn run() {
    #[cfg(all(
        feature = "wgpu",
        not(feature = "wgpu-spirv"),
        not(feature = "wgpu-msl")
    ))]
    {
        run_or_skip("wgpu<f32>", || {
            run_rms_norm::<cubecl::wgpu::WgpuRuntime, f32>(Default::default());
        });
    }

    #[cfg(feature = "wgpu-spirv")]
    {
        run_or_skip("wgpu-spirv<f16>", || {
            run_rms_norm::<cubecl::wgpu::WgpuRuntime, half::f16>(Default::default());
        });
    }

    #[cfg(feature = "wgpu-msl")]
    {
        run_or_skip("wgpu-msl<f32>", || {
            run_rms_norm::<cubecl::wgpu::WgpuRuntime, f32>(Default::default());
        });
    }

    #[cfg(feature = "cuda")]
    {
        use half::{bf16, f16};
        run_or_skip("cuda<f32>", || {
            run_rms_norm::<cubecl::cuda::CudaRuntime, f32>(Default::default());
        });
        run_or_skip("cuda<f16>", || {
            run_rms_norm::<cubecl::cuda::CudaRuntime, f16>(Default::default());
        });
        run_or_skip("cuda<bf16>", || {
            run_rms_norm::<cubecl::cuda::CudaRuntime, bf16>(Default::default());
        });
    }
}
