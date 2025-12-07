//! Row binning for load-balanced execution.
//!
//! The core innovation: group rows by nnz count so each bin has uniform work per row.
//! This enables perfect load balancing and strategy selection per bin.

use cubecl_core::ir::StorageType;
use cubecl_runtime::client::ComputeClient;
use cubecl_runtime::server::{ComputeServer, Handle};

use crate::error::SparseResult;
use crate::handle::{SparseTensorHandle, SparseTensorHandleRef};
use super::statistics::MatrixStatistics;
use crate::ops::spmm::config::BinningConfig;

/// Bin identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BinId(u8);

impl BinId {
    pub const EMPTY: BinId = BinId(0);
    pub const TINY: BinId = BinId(1);    // 1-7 nnz
    pub const SMALL: BinId = BinId(2);   // 8-31 nnz
    pub const MEDIUM: BinId = BinId(3);  // 32-127 nnz
    pub const LARGE: BinId = BinId(4);   // 128-511 nnz
    pub const HUGE: BinId = BinId(5);    // 512+ nnz

    /// Create from histogram index.
    pub fn from_index(idx: usize) -> Self {
        BinId(idx as u8)
    }

    /// Get nnz range for this bin.
    pub fn nnz_range(&self) -> core::ops::Range<u32> {
        match self.0 {
            0 => 0..1,
            1 => 1..8,
            2 => 8..32,
            3 => 32..128,
            4 => 128..512,
            5 => 512..u32::MAX,
            _ => unreachable!(),
        }
    }

    /// Get padded nnz for this bin (aligned for memory/TC).
    ///
    /// Padding ensures:
    /// - Aligned memory access
    /// - Tensor core compatibility (multiples of 16)
    /// - No branch divergence in kernels
    pub fn padded_nnz(&self) -> u32 {
        match self.0 {
            0 => 0,
            1 => 8,
            2 => 32,
            3 => 128,
            4 => 512,
            5 => 1024, // Or dynamically based on actual max
            _ => unreachable!(),
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self.0 {
            0 => "empty",
            1 => "tiny",
            2 => "small",
            3 => "medium",
            4 => "large",
            5 => "huge",
            _ => "unknown",
        }
    }
}

/// Execution strategy for a bin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinStrategy {
    /// Skip (empty rows).
    Skip,

    /// One thread per row.
    /// Best for: 1-7 nnz (TINY)
    RowSplit,

    /// One warp per row.
    /// Best for: 8-31 nnz (SMALL)
    WarpPerRow,

    /// Vectorized warp per row.
    /// Best for: 8-31 nnz (SMALL) with large N
    VectorWarpPerRow { vec_width: u32 },

    /// Gather B then dense GEMM (CUDA cores).
    /// Best for: 32-127 nnz (MEDIUM)
    GatherGemm { tile_n: u32 },

    /// Gather B then dense GEMM (Tensor cores).
    /// Best for: 128+ nnz (LARGE, HUGE) with tensor core support
    GatherTensorCore {
        tile_m: u32,
        tile_k: u32,
        tile_n: u32,
    },

    /// Merge-path for irregular distribution.
    /// Best for: high CV within bin
    MergePath,
}

/// A single bin containing rows with similar nnz.
pub struct RowBin {
    /// Bin identifier.
    pub id: BinId,
    
    /// Number of rows in this bin.
    pub num_rows: u32,
    
    /// Padded nnz (all rows padded to this).
    pub padded_nnz: u32,
    
    /// Total actual nnz across all rows.
    pub total_nnz: u64,
    
    /// Original row indices.
    pub row_indices: Handle,
    
    /// Flattened, padded column indices.
    /// Shape: [num_rows, padded_nnz]
    pub gather_cols: Handle,
    
    /// Flattened, padded values.
    /// Shape: [num_rows, padded_nnz]
    pub gather_vals: Handle,
    
    /// Selected execution strategy.
    pub strategy: BinStrategy,
}

/// Complete row binning for a matrix.
pub struct RowBinning

{
    /// Non-empty bins.
    pub bins: alloc::vec::Vec<RowBin>,

    /// Total rows across all bins.
    pub total_rows: u32,

    /// Total nnz across all bins.
    pub total_nnz: u64,
}

impl RowBinning

{
    /// Get bin by ID.
    pub fn get_bin(&self, id: BinId) -> Option<&RowBin> {
        self.bins.iter().find(|b| b.id == id)
    }

    /// Iterate over non-skip bins.
    pub fn executable_bins(&self) -> impl Iterator<Item = &RowBin> {
        self.bins.iter().filter(|b| b.strategy != BinStrategy::Skip)
    }
}
use crate::memory::pool::SparseBufferSet;
/// Create row binning from CSR matrix.
///
/// # Implementation
/// 1. Compute row lengths (or reuse from stats)
/// 2. Count rows per bin (histogram or scan)
/// 3. Allocate bin buffers (row_indices, gather_cols, gather_vals)
/// 4. Populate bins via compaction kernel:
///    - Scatter rows to bins based on length
///    - Flatten and pad column indices and values
/// 5. Select strategy per bin
///
/// # Algorithm Steps
/// ```text
/// Step 1: Compute row lengths
///   row_lengths[i] = row_ptrs[i+1] - row_ptrs[i]
///
/// Step 2: Count rows per bin
///   Parallel for i in 0..M:
///     bin = classify_row_length(row_lengths[i], boundaries)
///     atomic_add(&bin_counts[bin], 1)
///
/// Step 3: Compute bin offsets (prefix sum)
///   bin_offsets = exclusive_scan(bin_counts)
///
/// Step 4: Allocate bin buffers
///   For each bin b with count > 0:
///     row_indices[b] = allocate(bin_counts[b])
///     gather_cols[b] = allocate(bin_counts[b] * padded_nnz[b])
///     gather_vals[b] = allocate(bin_counts[b] * padded_nnz[b])
///
/// Step 5: Populate bins via compaction
///   Parallel for row in 0..M:
///     nnz = row_lengths[row]
///     bin = classify_row_length(nnz, boundaries)
///
///     // Atomically get position in bin
///     local_row = atomic_add(&bin_positions[bin], 1)
///
///     // Write row index
///     row_indices[bin][local_row] = row
///
///     // Flatten and pad column indices and values
///     row_start = row_ptrs[row]
///     row_end = row_ptrs[row + 1]
///     padded = padded_nnz[bin]
///
///     for i in 0..padded:
///       if i < nnz:
///         gather_cols[bin][local_row * padded + i] = col_indices[row_start + i]
///         gather_vals[bin][local_row * padded + i] = values[row_start + i]
///       else:
///         gather_cols[bin][local_row * padded + i] = 0  // padding
///         gather_vals[bin][local_row * padded + i] = 0.0
///
/// Step 6: Select strategy
///   For each bin, call select_bin_strategy based on:
///     - Bin ID (row length range)
///     - Number of rows in bin
///     - Output columns N
///     - Hardware capabilities (tensor cores)
/// ```
use crate::prelude::SparseFormatId;
pub fn create_binning<R: cubecl_runtime::runtime::Runtime>(
    sparse: SparseTensorHandleRef,
    stats: &MatrixStatistics,
    config: &BinningConfig,
    n_output_cols: u32,
    client: &ComputeClient<R>,
) -> SparseResult<RowBinning>

{
    let m = stats.rows;
    let nnz = stats.nnz;

    // Get buffer references
    let buffers = sparse.buffers;
    let (row_ptrs, col_indices, values) = match buffers {
        SparseBufferSet::Csr {
            row_ptrs,
            col_indices,
            values,
        } => (row_ptrs, col_indices, values),
        _ => {
            return Err(crate::error::SparseError::UnsupportedFormat {
                op: "create_binning",
                format: SparseFormatId::Csr,
            })
        }
    };

    // TODO: Step 1: Compute row lengths
    // let row_lengths = compute_row_lengths_kernel(row_ptrs, m, client);

    // TODO: Step 2: Count rows per bin
    // let bin_counts = count_rows_per_bin_kernel(&row_lengths, &config.boundaries, client);

    // TODO: Step 3: Prefix sum for offsets
    // let bin_offsets = exclusive_scan(&bin_counts, client);

    // TODO: Step 4 & 5: Allocate and populate bins
    // For each non-empty bin, launch compaction kernel

    // Placeholder: Create empty bins vector
    let bins = alloc::vec::Vec::new();

    // TODO: Step 6: Select strategies
    // For each bin, determine strategy based on characteristics

    Ok(RowBinning {
        bins,
        total_rows: m,
        total_nnz: nnz,
    })
}

/// Select execution strategy for a bin.
///
/// # Heuristics
/// - EMPTY: Skip
/// - TINY (1-7): RowSplit
/// - SMALL (8-31): WarpPerRow or VectorWarpPerRow if N ≥ 128
/// - MEDIUM (32-127): GatherGemm or GatherTC if has tensor cores and N ≥ 64
/// - LARGE/HUGE (128+): GatherTC if has tensor cores, else GatherGemm
/// - High CV: Override with MergePath
///
/// # Reasoning
/// ```text
/// Row length determines parallelism granularity:
///   1-7 nnz: Thread-level (no cooperation needed)
///   8-31 nnz: Warp-level (warp primitives efficient)
///   32-127 nnz: Block-level gather + GEMM (overhead amortized)
///   128+ nnz: Block-level gather + Tensor Cores (8-16× speedup)
///
/// Output width N determines tiling:
///   N < 64: Small tiles, CUDA cores sufficient
///   N ≥ 64: Tensor cores beneficial (16×16 fragments)
///   N ≥ 128: Vectorization beneficial
/// ```
pub fn select_bin_strategy(
    bin_id: BinId,
    num_rows: u32,
    n_output_cols: u32,
    has_tensor_cores: bool,
    config: &BinningConfig,
) -> BinStrategy {
    match bin_id {
        BinId::EMPTY => BinStrategy::Skip,

        BinId::TINY => BinStrategy::RowSplit,

        BinId::SMALL => {
            if n_output_cols >= 128 {
                BinStrategy::VectorWarpPerRow { vec_width: 4 }
            } else {
                BinStrategy::WarpPerRow
            }
        }

        BinId::MEDIUM => {
            if has_tensor_cores && config.enable_tensor_cores && n_output_cols >= 64 {
                BinStrategy::GatherTensorCore {
                    tile_m: 16,
                    tile_k: 32,
                    tile_n: 64,
                }
            } else if num_rows >= config.min_rows_for_gather {
                BinStrategy::GatherGemm { tile_n: 64 }
            } else {
                BinStrategy::WarpPerRow
            }
        }

        BinId::LARGE | BinId::HUGE => {
            if has_tensor_cores && config.enable_tensor_cores && n_output_cols >= 128 {
                BinStrategy::GatherTensorCore {
                    tile_m: 16,
                    tile_k: 64,
                    tile_n: 128,
                }
            } else {
                BinStrategy::GatherGemm { tile_n: 128 }
            }
        }

        _ => BinStrategy::WarpPerRow,
    }
}
