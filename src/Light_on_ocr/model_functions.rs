use candle_core::Device;
use crate::{Light_on_ocr::{config_structs::ModelConfig, model::LightOnOCR}, *};

const IM_START:   u32 = 151644; // <|im_start|>
const IM_END:     u32 = 151645; // <|im_end|> — EOS token
const VISION_END: u32 = 151653; // <|vision_end|> — terminates image sequence
const VISION_PAD: u32 = 151654; // <|vision_pad|> — row break between patch rows
const IMAGE_PAD:  u32 = 151655; // <|image_pad|> — one image patch token


/*Selects the device to you, falls back to cpu if no CUDA or METAL devices found */


pub fn build_model(device: &Device) -> Result<(LightOnOCR, Tokenizer)> {
    
    let config_path    = "models/LightOnOCR/config.json";
    let weights_path   = "models/LightOnOCR/model.safetensors";
    let tokenizer_path = "models/LightOnOCR/tokenizer.json";

    let config_str = std::fs::read_to_string(config_path)?;
    let model_config: ModelConfig = serde_json::from_str(&config_str)?;

    let dtype = get_dtype(device);

    println!("Loading weights");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
    }?;

    println!("Building model");
    let mut model = LightOnOCR::new(&model_config, vb)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))?;

    Ok((model, tokenizer))
}

/*image path is hardcoded for now should later use the pages vector. */
pub fn run_model(mut model: LightOnOCR, tokenizer: Tokenizer, device: Device, pages: Vec<Page>) -> Result<()> {
    
    let image_path = "data/images/pol-1986-03-04-L854-Anja-Landgree-Inga-Ålenius.pdf-23.png";
    let img = image::open(image_path)?;
    let preprocessed = preprocess(&img, &device)?;

    // Merged patch grid dimensions after 2x2 spatial merge
    let merged_ph = preprocessed.ph / 2;
    let merged_pw = preprocessed.pw / 2;
    let num_image_tokens = merged_ph * merged_pw; // IMAGE_PAD count = 2200

    println!("ph={} pw={} merged_ph={} merged_pw={} num_image_tokens={}",
        preprocessed.ph, preprocessed.pw, merged_ph, merged_pw, num_image_tokens);

    // Encode only plain text — special tokens inserted by id
    let encode = |s: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .get_ids()
            .to_vec())
    };

    let system_tokens    = encode("system")?;
    let user_tokens      = encode("user\n")?;
    let assistant_tokens = encode("assistant\n")?;
    let newline_tokens   = encode("\n")?;

    let mut image_tokens: Vec<u32> = Vec::with_capacity(num_image_tokens + merged_ph);
    for row in 0..merged_ph {
        for _ in 0..merged_pw {
            image_tokens.push(IMAGE_PAD);
        }
        if row < merged_ph - 1 {
            image_tokens.push(VISION_PAD);
        }
    }
    image_tokens.push(VISION_END);

    // Full prompt:
    // <|im_start|>system<|im_end|>\n
    // <|im_start|>user\n[image tokens]<|im_end|>\n
    // <|im_start|>assistant\n
    let mut input_ids: Vec<u32> = Vec::new();

    input_ids.push(IM_START);
    input_ids.extend_from_slice(&system_tokens);
    input_ids.push(IM_END);
    input_ids.extend_from_slice(&newline_tokens);

    input_ids.push(IM_START);
    input_ids.extend_from_slice(&user_tokens);
    input_ids.extend_from_slice(&image_tokens);
    input_ids.push(IM_END);
    input_ids.extend_from_slice(&newline_tokens);

    input_ids.push(IM_START);
    input_ids.extend_from_slice(&assistant_tokens);

    let seq_len = input_ids.len();
    println!("Sequence length: {} ({} IMAGE_PAD + {} row tokens)",
        seq_len, num_image_tokens, merged_ph);

    let input_tensor = candle_core::Tensor::from_vec(
        input_ids,
        (1, seq_len),
        &device,
    )?;

    println!("Prefilling...");
    let logits = model.forward(&input_tensor, &preprocessed.pixel_values, 0)?;
    println!("logits shape: {:?}", logits.shape());

    let mut generated: Vec<u32> = Vec::new();
    let mut offset = seq_len;

    let first_token = greedy(&logits)?;
    generated.push(first_token);
    println!("first token id={} decoded={:?}",
        first_token,
        tokenizer.decode(&[first_token], false));

    println!("Generating...");
    let max_new_tokens = 1024usize;

    for _ in 1..max_new_tokens {
        let last = *generated.last().unwrap();

        if last == IM_END {
            break;
        }

        let input = candle_core::Tensor::from_vec(
            vec![last],
            (1, 1),
            &device,
        )?;

        let logits = model.decode_step(&input, offset)?;
        let token = greedy(&logits)?;
        generated.push(token);
        offset += 1;
    }

    let decode_ids: Vec<u32> = generated.iter()
        .copied()
        .filter(|&t| t != IM_END)
        .collect();

    let output = tokenizer
        .decode(&decode_ids, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("\n=== Output ===");
    println!("{}", output);

    Ok(())
}


fn greedy(logits: &candle_core::Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?;
    let seq = logits.dim(0)?;
    let last = logits.narrow(0, seq - 1, 1)?.squeeze(0)?;
    Ok(last.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?)
}



pub fn print_safetensors() -> Result<()> {
    let tensor1 = "models/models--lightonai--LightOnOCR-2-1B-bbox-soup/snapshots/dfdbd3e3627d80e28ddadece14098131aa485700/model.safetensors";
    
    let tensors = load(tensor1, &candle_core::Device::Cpu)?;

    let mut out = std::fs::File::create("Keys.txt")?;
    
    for (name, tensor) in &tensors{
        writeln!(out, "{}\t{:?}\t{:?}", name, tensor.shape(), tensor.dtype())?;
    }

    Ok(())
}