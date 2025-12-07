//! Pruning strategies

#[derive(Clone, Debug)]
pub enum PruningStrategy {
    MagnitudeGlobal { target_sparsity: f32 },
    MagnitudeStructured { target_sparsity: f32, dim: usize },
    Random { target_sparsity: f32, seed: u64 },
    NMStructured { n: usize, m: usize },
    Block { target_sparsity: f32, block_rows: usize, block_cols: usize },
}

pub struct Pruner;

impl Pruner {
    pub fn prune() {
        todo!()
    }
}
