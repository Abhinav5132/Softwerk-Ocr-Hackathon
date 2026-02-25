use candle_core::{D, DType, Device, Shape, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, Module, VarBuilder, conv2d_no_bias};
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
        let rms = (&rms + self.eps)?.sqrt()?;

        let t_norm = t.broadcast_div(&rms)?;
        
        // Reshape weight to have compatible shape for broadcasting
        let weight = self.weight.reshape(vec![self.weight.dim(0)? as usize])?;
        Ok(t_norm.broadcast_mul(&weight)?)
    }
}

pub struct visionRotaryEmbedding{
    rope_theta: u32,
    head_dim: u32
}
impl visionRotaryEmbedding{

    pub fn new(cfg: &VisionConfig) -> Result<Self> {
        Ok(
            Self { rope_theta: cfg.rope_theta, head_dim: cfg.head_dim }
        )
    }

    pub fn freqs(&self, h: usize, w: usize, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let half_dim = self.head_dim/2;

        let inv_freq: Vec<f32> = (0..half_dim / 2)
        .map(|i| 1.0 / (self.rope_theta as f32).powf(2.0 * i as f32 / half_dim as f32))
        .collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?
        .to_dtype(dtype)?;

        let row_ids: Vec<f32> = (0..h).flat_map(|r| vec![r as f32; w]).collect();
        let col_ids: Vec<f32> = (0..h).flat_map(|_| (0..w).map(|c| c as f32).collect::<Vec<_>>()).collect();

        let row_ids = Tensor::new(row_ids.as_slice(), device)?
            .to_dtype(dtype)?;
        let col_ids = Tensor::new(col_ids.as_slice(), device)?
            .to_dtype(dtype)?;

        let row_angles = row_ids.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        let col_angles = col_ids.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        
        let angles = Tensor::cat(&[&row_angles, &col_angles], 1)?.to_dtype(dtype)?;

        Ok((angles.cos()?, angles.sin()?))
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(candle_core::D::Minus1)?;
    let half = d / 2;

    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;

    Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
}

pub fn apply_rotary_embedding(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    //x: (n_heads, seq_len, head_dim)
    // cos, sin: (seq_len, head_dim) - needs to be unsqueezed to (1, seq_len, head_dim) for broadcasting

    let cos = Tensor::cat(&[cos, cos], candle_core::D::Minus1)?; //(seq_len, head_dim)
    let sin = Tensor::cat(&[sin, sin], candle_core::D::Minus1)?;

    let cos = cos.unsqueeze(0)?;  // (1, seq_len, head_dim)
    let sin = sin.unsqueeze(0)?;  // (1, seq_len, head_dim)

    let rotated = rotate_half(x)?;

    Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?) // THIS IS HORRENDUS DO ACTUAL ERROR HANDELING LATER
}

pub struct VisionAttention {
    q_proj: Linear, //query
    k_proj: Linear, //key
    v_proj: Linear, //value
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention{

    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size as usize;

        let q_proj = candle_nn::linear_no_bias(h, h, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(h, h, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(h, h, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(h, h, vb.pp("o_proj"))?;

        let scale = 1.0 / (cfg.head_dim as f64).sqrt();

        Ok(
            Self { q_proj, k_proj, v_proj, o_proj, num_heads: cfg.num_attention_heads as usize, head_dim: cfg.head_dim as usize, scale }
        )
    }

    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        //project to q, k, v
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?; 
        let v = self.v_proj.forward(x)?; 

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?
        .transpose(1, 2)?;
        let k = k.reshape((b, seq_len, self.num_heads, self.head_dim))?
        .transpose(1, 2)?;
        let v = v.reshape((b, seq_len, self.num_heads, self.head_dim))?
        .transpose(1, 2)?;

        //apply RoPE 
        let q = q.reshape((b * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((b * self.num_heads, seq_len, self.head_dim))?;

        let q = apply_rotary_embedding(&q, cos, sin)?;
        let k = apply_rotary_embedding(&k, cos, sin)?;

        let q = q.reshape((b, self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((b, self.num_heads, seq_len, self.head_dim))?;
        
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let scores = q.matmul(&k.transpose(2, 3)?)?.affine(self.scale, 0.0)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = weights.matmul(&v)?;

        let out = out.transpose(1, 2)?.reshape((b, seq_len, self.num_heads * self.head_dim))?;

        Ok(self.o_proj.forward(&out)?)
    }
}

pub struct visionMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear
}

impl visionMlp {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = candle_nn::linear_no_bias(cfg.hidden_size as usize, cfg.intermediate_size as usize, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(cfg.hidden_size as usize, cfg.intermediate_size as usize, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(cfg.intermediate_size as usize, cfg.hidden_size as usize, vb.pp("down_proj"))?;

        Ok(Self { gate_proj, up_proj, down_proj })

    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor>{
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;

        let hidden = (gate * up)?;
        Ok(self.down_proj.forward(&hidden)?)
    }
}

pub struct visionLayer {
    attention_norm: RmsNorm,
    attention: VisionAttention,
    ffn_norm: RmsNorm,
    mlp: visionMlp,
}

impl visionLayer {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let attention_norm = RmsNorm::new(cfg.hidden_size as usize, 1e-5, vb.pp("attention_norm"))?;
        let attention = VisionAttention::new(cfg, vb.pp("attention"))?;
        let ffn_norm = RmsNorm::new(cfg.hidden_size as usize, 1e-5, vb.pp("ffn_norm"))?;
        let mlp = visionMlp::new(cfg, vb.pp("feed_forward"))?;

        Ok(
            visionLayer { attention_norm, attention, ffn_norm, mlp }
        )
    }

    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.attention_norm.forward(x)?;
        let x = self.attention.forward(&x, cos, sin)?;
        let x = (residual + x)?;

        let residual = &x;
        let x_normed = self.ffn_norm.forward(&x)?;
        let x_mlp = self.mlp.forward(&x_normed)?;
        Ok((residual + x_mlp)?)
    }
}

//this is what actually turns images into patch embeddings
pub struct visionEncoder{
    patch_conv: Conv2d,
    ln_pre: RmsNorm,
    layers: Vec<visionLayer>,
    rope: visionRotaryEmbedding,
    pub(crate) patch_size: usize,
}

impl visionEncoder {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_config = Conv2dConfig{
            stride: cfg.patch_size as usize,
            ..Default::default()
        };

        let patch_conv = conv2d_no_bias(
            cfg.num_channels as usize, 
            cfg.hidden_size as usize, 
            cfg.patch_size as usize, 
            conv_config, 
            vb.pp("patch_conv"),)?;

        let ln_pre = RmsNorm::new(cfg.hidden_size as usize, 1e-5, vb.pp("ln_pre"))?;

        let transformer_vb = vb.pp("transformer").pp("layers");

        let layers = (0..cfg.num_hidden_layers)
        .map(|i| visionLayer::new(cfg, transformer_vb.pp(i)))
        .collect::<Result<Vec<_>>>()?;
        
        let rope = visionRotaryEmbedding::new(cfg)?;

        Ok(Self { patch_conv, ln_pre, layers, rope, patch_size: cfg.patch_size as usize })

    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = pixel_values.dims4()?;

        let x = self.patch_conv.forward(pixel_values)?;

        let (b, c, ph, pw) = x.dims4()?;
        println!("conv output: ph={} pw={}", ph, pw);
        let x = x.reshape((b, c, ph*pw))?.transpose(1, 2)?;

        let x = self.ln_pre.forward(&x)?;

        let device = pixel_values.device();
        let d_type = pixel_values.dtype();
        let (cos, sin) = self.rope.freqs(ph, pw, device, d_type)?;

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin)?;
        }

        Ok(x.squeeze(0)?) // return (num_of_patches, hidden_dim)
    }
}