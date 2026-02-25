use std::str::FromStr;

use candle_core::DType;
use candle_nn::Activation;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub dtype: String,
    pub eos_token_id: u32,
    pub image_token_id: u32,
    pub model_type: String,
    pub multimodal_projector_bias: bool,
    pub pad_token_id: u32,
    pub projector_hidden_act: String,
    pub spatial_merge_size: u32,
    pub text_config: TextConfig,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vision_config: VisionConfig,
    pub vision_feature_layer: i32
  
}

#[derive(Deserialize)]
pub struct TextConfig{
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: usize,
    pub dtype: String,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub rope_theta: usize,
    pub sliding_window: Option<serde_json::Value>,
    pub use_cache: bool,
    pub use_sliding_window: bool,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
    pub use_qk_norm: bool
}

impl TextConfig {
    pub fn get_activation(&self) -> Activation {
        return Activation::Silu
    }
}

#[derive(Deserialize)]
pub struct RopeParameters {
    rope_theta: u32,
    rope_type: String,
}

#[derive(Deserialize)]
pub struct VisionConfig{
    pub attention_dropout: u32,
    pub dtype: String,
    pub head_dim: u32,
    pub hidden_act: String,
    pub hidden_size: u32,
    pub image_size: u32,
    pub initializer_range: f32,
    pub intermediate_size: u32,
    pub model_type: String,
    pub num_attention_heads: u32,
    pub num_channels: u32,
    pub num_hidden_layers: u32,
    pub patch_size: u32,
    pub rope_parameters: RopeParameters,
    pub rope_theta: u32
}

impl VisionConfig {
    /*Image gets cut into a grid of smaller Patches. This returns the number of patches */
    pub fn num_patches(&self) -> u32 {
        (self.image_size / self.patch_size).pow(2) 
    }

    /*Size of each attention head */
    pub fn head_dim(&self) -> u32 {
        self.hidden_size / self.num_attention_heads
    }
}