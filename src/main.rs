use std::fs;
use std::process::Command;

use rayon::prelude::*;

fn main() {
    let paths: Vec<_> = fs::read_dir("./data")
        .expect("failed to read data directory")
        .into_iter()
        .map(|ent| ent.expect("failed to get path"))
        .filter(|ent| ent.file_type().expect("").is_file())
        .map(|ent| ent.path())
        .collect();

    paths.into_par_iter().for_each(|path| {
        let name = path.as_path().file_name().unwrap().display();
        let _ = Command::new("pdfimages")
            .arg("-j")
            .arg(path.as_os_str())
            .arg(format!("./data/images/{name}"))
            .spawn();
    });
}
