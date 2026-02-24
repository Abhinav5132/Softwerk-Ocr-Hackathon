use candle_core::{D, DType, Device, Shape, Tensor};
use candle_nn::{Conv2d, VarBuilder};
use anyhow::Result;

use crate::configStructs::VisionConfig;

pub struct RmsNorm{
    weight: Tensor,
    eps: f64
}

impl RmsNorm{
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t_sq = t.sqr()?;
        let rms = t_sq.mean_keepdim(D::Minus1)?;
        let rms = (rms + self.eps)?.sqrt()?;

        let t_norm = t.broadcast_div(&rms)?;

        Ok(t_norm.broadcast_mul(&self.weight)?)
    }
}

impl VisionConfig{
    pub fn freqs(&self, h: usize, w: usize, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let half_dim = self.head_dim()/2;

        let inv_freq: Vec<u32> = (0..half_dim / 2)
        .map(|i| 1 / self.rope_theta.pow(2 * i / half_dim)).collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?
        .to_dtype(dtype)?;

        let row_ids: Vec<f32> = (0..h).flat_map(|r| vec![r as f32; w]).collect();
        let col_ids: Vec<f32> = (0..h).flat_map(|_| (0..w).map(|c| c as f32).collect::<Vec<_>>()).collect();

        let row_ids = Tensor::new(row_ids.as_slice(), device)?;
        let col_ids = Tensor::new(col_ids.as_slice(), device)?;

        let row_angles = row_ids.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        let col_angles = col_ids.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        
        let angles = Tensor::cat(&[&row_angles, &col_angles], 1)?.to_dtype(dtype)?;

        Ok((angles.cos()?, angles.sin()?))
    }
}