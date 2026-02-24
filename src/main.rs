use std::fs::{self, remove_dir_all, remove_file};
use std::path::Path;
use std::process::Command;
use std::time;
use std::io::Write;
use rayon::prelude::*;
use anyhow::Result;
use candle_core::safetensors::*;

mod configStructs;
mod vision_encoder;
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

   let _ = print_safetensors();
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