use candle_nn::Activation;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub dtype: String,
    pub eos_token_id: usize,
    pub image_token_id: usize,
    pub model_type: String,
    pub multimodal_projector_bias: bool,
    pub pad_token_id: usize,
    pub projector_hidden_act: String,
    pub spatial_merge_size: usize,
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
        Activation::Silu 
    }
}

#[derive(Deserialize)]
pub struct RopeParameters {
    rope_theta: usize,
    rope_type: String,
}

#[derive(Deserialize)]
pub struct VisionConfig{
    pub attention_dropout: usize,
    pub dtype: String,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub image_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    pub rope_parameters: RopeParameters,
    pub rope_theta: usize
}
