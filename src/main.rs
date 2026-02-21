use std::{ffi::OsString, fs, path::PathBuf};

use pdfium_render::prelude::*;
use anyhow::{Result, anyhow};
fn main() -> Result<()> {
    render_pdf_to_png()?;
    Ok(())
}


pub fn render_pdf_to_png() -> Result<()> {
    let pdfium = Pdfium::new(
        Pdfium::bind_to_library(
            Pdfium::pdfium_platform_library_name_at_path(
            "extern/lib/"
        ))?
    );

    let data_location = "data/raw"; //TODO: make this dynamic and add error handling for that 

    for entry in fs::read_dir(data_location)? {
        let entry = match entry {
            Ok(file) => file, 
            Err(e) => {
                dbg!(e);
                continue; // try to convert next file if this one fails 
            }
        };
        let filename = match entry.file_name().into_string(){
            Ok(name) => name,
            Err(e) => {
                dbg!(e);
                "unknown".to_string()
            }
        };
        let path = entry.path();

        let doc = match Pdfium::load_pdf_from_file(&pdfium, &path, None) {
            Ok(d) => d, 
            Err(e) => {
                dbg!(&e);
                return Err(anyhow::Error::new(e));
            }
        };
        // TODO clean already converted before trying to convert

        /*match extract_images_from_pdf(doc, filename) {
            Ok(_) => {
                continue;
            }

            Err(e) => {
                dbg!(e);
                continue;
            }
        }*/
        
        match convert_to_png(doc, filename) {
            Ok(_) => {
                continue;
            }
            Err(e) => {
                dbg!(e);
                continue;
            }
        }
    }
    Ok(())
}

pub fn extract_images_from_pdf(doc: PdfDocument, filename: String) -> Result<()> {
    let pages = doc.pages();
    if pages.is_empty(){
        return Ok(());
    }

    for (index, page) in pages.iter().enumerate(){
        for (img_index, img) in page.objects().iter().enumerate(){
            if let Some(image) = img.as_image_object(){
                let bitmap = image.get_raw_bitmap()?;

                let img_name = format!("data/converted/{filename}_page_{index}_{img_index}.png");
                bitmap.as_image().save(&img_name)?;
                dbg!(img_name);
            }
        }
    }  
    Ok(())  
}


pub fn convert_to_png(doc: PdfDocument, filename: String) -> anyhow::Result<()> {

    let pages = doc.pages();
    if pages.is_empty() {
        return Ok(());
    }

    for (index, page )in pages.iter().enumerate(){
        let render = page.render_with_config(
            &PdfRenderConfig::new().scale_page_by_factor(2.0) // this can be turned up higher if the models need higher quality image
        );

        let bitmap = render?.as_image();
        let image_name=format!("data/converted/{filename}_page_{index}.png");

        bitmap.save(&image_name)?;
        dbg!(image_name);
    }

    Ok(())
}