//! Custom reduce operations for fused kernels.
//!
//! These reducers fuse element-wise transformations with reduction operations
//! to minimize kernel launches and maximize GPU efficiency.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_reduce::ReducePrecision;
use cubecl_reduce::instructions::{ReduceCoordinate, ReduceFamily, ReduceInstruction, ReduceRequirements};

/// Reduce by summing squared values: sum(x_i^2)
///
/// This fuses the square operation with the sum reduction, eliminating
/// the need for a separate square_kernel launch.
///
/// **Performance**: 1 kernel launch instead of 2 separate operations.
#[derive(Debug, CubeType, Clone)]
pub struct SumSquared;

impl ReduceFamily for SumSquared {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for SumSquared {
    type AccumulatorItem = Line<P::EA>;
    type SharedAccumulator = SharedMemory<Line<P::EA>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn from_config(_config: Self::Config) -> Self {
        SumSquared {}
    }

    fn null_input(_this: &Self, #[comptime] line_size: u32) -> Line<P::EI> {
        Line::empty(line_size).fill(P::EI::from_int(0))
    }

    fn null_accumulator(_this: &Self, #[comptime] line_size: u32) -> Self::AccumulatorItem {
        Line::empty(line_size).fill(P::EA::from_int(0))
    }

    fn assign_accumulator(
        _this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        *destination = *source;
    }

    fn reduce(
        _this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Line<P::EI>,
        _coordinate: ReduceCoordinate,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        // Square the input inline during reduction
        let squared = item * item;

        if use_planes {
            *accumulator + plane_sum(Line::cast_from(squared))
        } else {
            *accumulator + Line::cast_from(squared)
        }
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        lhs + rhs
    }

    fn merge_line<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Out {
        let mut sum = P::EA::from_int(0);
        #[unroll]
        for k in 0..accumulator.size() {
            sum += accumulator[k];
        }
        Out::cast_from(sum)
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator)
    }
}
