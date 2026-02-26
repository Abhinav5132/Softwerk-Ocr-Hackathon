use std::fs::{self, remove_dir_all, remove_file};
use std::path::Path;
use std::process::Command;
use std::time;
use std::io::Write;
use candle_nn::VarBuilder;
use rayon::prelude::*;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, safetensors::*};
use tokenizers::Tokenizer;

use crate::configStructs::ModelConfig;
use crate::model::LightOnOCR;
use crate::preprocess::preprocess;

mod model;
mod configStructs;
mod projector;
mod language_models;
mod preprocess;
fn main() {
    /* 
    let start_time = time::Instant::now();
    let dir = "./data/images";

    if !Path::new(dir).exists() {
        fs::create_dir_all(dir).expect("failed to create dir");
    } else{
        fs::read_dir(dir).expect("failed to read images directory")
        .map(|entry| entry.expect("failed to get dirEntry"))
        .filter(|ent| ent.file_type().expect("unable to get file type").is_file())
        .for_each(|ent| { fs::remove_file(ent.path()).expect("failed to remove file"); });
    }

    let paths: Vec<_> = fs::read_dir("./data")
        .expect("failed to read data directory")
        .map(|ent| ent.expect("failed to get path"))
        .filter(|ent| ent.file_type().expect("").is_file())
        .map(|ent| ent.path())
        .collect();

    paths.into_par_iter().for_each(|path| {
        let name = path.as_path().file_name().unwrap().display();
        let _ = Command::new("pdftoppm")
            .arg("-png")
            .arg("-r").arg("200")
            .arg(path.as_os_str())
            .arg(format!("./data/images/{name}"))
            .status().unwrap_or_else(|_| panic!("failed to convert to png {name}"));
    });
    let elapsed = start_time.elapsed().as_secs();
    println!("{elapsed}");

    */

    match run_model() {
        Ok(_) => {},
        Err(e) => {
            dbg!(e);
        }
    };
}


const IM_START:   u32 = 151644; // <|im_start|>
const IM_END:     u32 = 151645; // <|im_end|> — EOS token
const VISION_END: u32 = 151653; // <|vision_end|> — terminates image sequence
const VISION_PAD: u32 = 151654; // <|vision_pad|> — row break between patch rows
const IMAGE_PAD:  u32 = 151655; // <|image_pad|> — one image patch token

pub fn run_model() -> Result<()> {
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else if candle_core::utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}", device);

    let config_path    = "models/LightOnOcr/snapshots/dfdbd3e3627d80e28ddadece14098131aa485700/config.json";
    let weights_path   = "models/LightOnOcr/snapshots/dfdbd3e3627d80e28ddadece14098131aa485700/model.safetensors";
    let tokenizer_path = "models/LightOnOcr/snapshots/dfdbd3e3627d80e28ddadece14098131aa485700/tokenizer.json";

    let config_str = std::fs::read_to_string(config_path)?;
    let model_config: ModelConfig = serde_json::from_str(&config_str)?;

    let dtype = match &device {
        Device::Cpu => DType::F32,
        _ => DType::BF16,
    };

    println!("Loading weights");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)
    }?;

    println!("Building model");
    let mut model = LightOnOCR::new(&model_config, vb)?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))?;

    let image_path = "data/images/730501a_sundbyberg_stockholm.pdf-01.png";
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

    // Build image token block with row breaks.
    // Each row: [IMAGE_PAD x merged_pw] [VISION_PAD]
    // Last row: [IMAGE_PAD x merged_pw] [VISION_END]
    // VISION_PAD = row separator, tells model where each patch row ends.
    // VISION_END = image terminator.
    // The splice only replaces IMAGE_PAD tokens — VISION_PAD and VISION_END
    // pass through as their own learned embeddings.
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

// Greedy decode: pick token with highest logit.
// Handles prefill shape (1, seq, vocab) and decode shape (1, 1, vocab).
fn greedy(logits: &candle_core::Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?;
    let seq = logits.dim(0)?;
    let last = logits.narrow(0, seq - 1, 1)?.squeeze(0)?;
    Ok(last.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?)
}



pub fn print_safetensors() -> Result<()> {
    let tensor1 = "models/models--lightonai--LightOnOCR-2-1B-bbox-soup/snapshots/dfdbd3e3627d80e28ddadece14098131aa485700/model.safetensors";
    
    let mut tensors = load(tensor1, &candle_core::Device::Cpu)?;


    let mut out = std::fs::File::create("Keys.txt")?;
    
    for (name, tensor) in &tensors{
        writeln!(out, "{}\t{:?}\t{:?}", name, tensor.shape(), tensor.dtype())?;
    }

    Ok(())
}