use candle_core::{IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::{configStructs::ModelConfig, language_models::{ModelForCausalLM}, projector::Projector, vision_encoder::visionEncoder};
use anyhow::Result;

pub struct LightOnOCR{
    pub vision_encoder: visionEncoder,
    pub projector: Projector,
    pub language_model: ModelForCausalLM,
    pub image_token_id: u32,
}

impl LightOnOCR {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let model_vb = vb.pp("model");

        let vision_encoder = visionEncoder::new(
            &cfg.vision_config, 
            model_vb.pp("vision_encoder")
        )?;

        let projector = Projector::new(
            (cfg.vision_config.hidden_size) as usize, 
            model_vb.pp("vision_projection")
        )?;

        let language_model = ModelForCausalLM::new(
            &cfg.text_config, 
            model_vb.pp("language_model")
        )?;

        Ok(Self { vision_encoder, projector, language_model, image_token_id: cfg.image_token_id })
    }

    pub fn forward(&mut self, input_ids: &Tensor, pixel_values:&Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, h, w) = pixel_values.dims4()?;
        println!("pixel_values shape: {:?}, dtype: {:?}", pixel_values.shape(), pixel_values.dtype());
        
        // Check if image has non-zero values
        let sum_val = pixel_values.sum_all()?.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?;
        println!("pixel_values sum: {}", sum_val);
        
        let ph = h / self.vision_encoder.patch_size;
        let pw = w / self.vision_encoder.patch_size;
        println!("model.rs computed: ph={} pw={}", ph, pw);
        let mut embeds = self.language_model.base.embed_tokens.forward(input_ids)?;

        let image_features = self.vision_encoder.forward(pixel_values)?;
        println!("image_features shape: {:?}, dtype: {:?}", image_features.shape(), image_features.dtype());

        let image_embeds = self.projector.forward(&image_features, ph, pw)?;
        println!("image_embeds after projector shape: {:?}, dtype: {:?}", image_embeds.shape(), image_embeds.dtype());
        
        let image_embeds = image_embeds.to_dtype(embeds.dtype())?;

        println!("embeds dtype: {:?}", embeds.dtype());
        println!("image_embeds dtype: {:?}", image_embeds.dtype());
        println!("embeds shape: {:?}", embeds.shape());
        println!("image_embeds shape: {:?}", image_embeds.shape());
        
        embeds = self.splice_image_embeddings(input_ids, &embeds, &image_embeds)?;

        Ok((self.language_model.forward_embeds(&embeds, offset))?)
    }

    pub fn splice_image_embeddings(&self, input_ids: &Tensor, embeds: &Tensor, image_embeds: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let num_image_tokens = image_embeds.dim(0)?;
        
        let ids: Vec<u32> = input_ids
            .squeeze(0)?
            .to_dtype(candle_core::DType::U32)?
            .to_vec1()?;

        let image_positions: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == self.image_token_id)
            .map(|(i, _)| i)
            .collect();

        println!("Image token ID from config: {}", self.image_token_id);
        println!("Number of IMAGE_PAD tokens in input_ids: {}", image_positions.len());
        println!("Number of image embeddings from projector: {}", num_image_tokens);

        if image_positions.len() != num_image_tokens {
            panic!(
                "input_ids has {} image token positions but projector output has {} tokens",
                image_positions.len(),
                num_image_tokens
            );
        }

        let hidden = embeds.dim(2)?;
        let mut rows: Vec<Tensor> = Vec::with_capacity(seq_len);
        let mut img_idx = 0usize;

        for i in 0..seq_len {
            if ids[i] == self.image_token_id {
                let row = image_embeds.i(img_idx)?.unsqueeze(0)?;
                rows.push(row);
                img_idx += 1;
            } else {
                let row = embeds.i((0, i))?.unsqueeze(0)?;
                rows.push(row);
            }
        }
        Ok(Tensor::cat(&rows, 0)?
            .unsqueeze(0)?)  
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }

    pub fn decode_step(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        Ok(self.language_model.forward(input_ids, offset)?)
    }
}