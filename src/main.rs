use std::fs::{self};
use std::path::Path;
use std::process::Command;
use std::time;
use std::io::Write;
use candle_nn::VarBuilder;
use rayon::prelude::*;
use anyhow::Result;
use candle_core::{DType, safetensors::*};
use tokenizers::Tokenizer;

pub mod Light_on_ocr;
pub mod trocr;
use crate::Light_on_ocr::model_functions::{build_model, run_model, select_device};

mod page_struct;
use crate::page_struct::Page;
use crate::Light_on_ocr::preprocess::preprocess;

fn main() {
    
    let dir = "./data/images";
    let start_time = time::Instant::now();

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
    
    let device = select_device();
    let pages = vec![]; // Empty vector for now add actual loading later,
    if let Ok((model, tokenizer)) = build_model(&device){
        match run_model(model, tokenizer, device, pages) {
            Ok(_) => {
                println!("Transcription finished successfully")
            }

            Err(e) => {
                dbg!(e);
                println!("Failed to transcribe. Please fix the error above.")
            }
        }
    }
    else {
        println!("Failed to build model");
    }
    let elapsed = start_time.elapsed().as_secs();
    println!("{elapsed}");

}



