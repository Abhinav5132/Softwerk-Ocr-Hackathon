use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::{trocr, vit};
use tokenizers::Tokenizer;
use anyhow::Result;

use crate::trocr::config_structs::ModelConfig;
pub mod config_structs;

pub fn build_handwritten_trocr(config: ModelConfig, weights_path: &str, tokenizer: Tokenizer, device: &Device, dtype: DType) -> Result<trocr::TrOCRModel> {
    let encoder = config.encoder;
    let encoder_config = vit::Config{
        hidden_size: encoder.hidden_size,
        num_hidden_layers: encoder.num_hidden_layers,
        num_attention_heads: encoder.num_attention_heads,
        intermediate_size: encoder.intermediate_size,
        hidden_act: candle_nn::Activation::Gelu,
        layer_norm_eps: encoder.layer_norm_eps,
        image_size: encoder.image_size,
        patch_size: encoder.patch_size,
        num_channels: encoder.num_channels,
        qkv_bias: encoder.qkv_bias,
    };

    let decoder = config.decoder;
    let decoder_config = trocr::TrOCRConfig{
        vocab_size: decoder.vocab_size,
        d_model: decoder.d_model,
        cross_attention_hidden_size: decoder.cross_attention_hidden_size,
        decoder_layers: decoder.decoder_layers,
        decoder_attention_heads: decoder.decoder_attention_heads,
        decoder_ffn_dim: decoder.decoder_ffn_dim,
        activation_function: candle_nn::Activation::Relu,
        max_position_embeddings: decoder.max_position_embeddings,
        dropout: decoder.dropout,
        attention_dropout: decoder.attention_dropout,
        activation_dropout: decoder.activation_dropout,
        decoder_start_token_id: decoder.decoder_start_token_id,
        init_std: decoder.init_std,
        decoder_layerdrop: decoder.decoder_layerdrop,
        use_cache: decoder.use_cache,
        scale_embedding: decoder.scale_embedding,
        pad_token_id: decoder.pad_token_id,
        bos_token_id: decoder.bos_token_id,
        eos_token_id: decoder.eos_token_id,
        decoder_vocab_size: Some(decoder.vocab_size),
        use_learned_position_embeddings: decoder.use_learned_position_embeddings,
        tie_word_embeddings: decoder.tie_word_embeddings,
    };

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
    }?;
    let mut model = trocr::TrOCRModel::new(&encoder_config, &decoder_config, vb)?;
    Ok(model)
}
