

/*
path -> path to the PNG file
transcription type -> the result of transcription type classifier
img_coordinates -> if there is an image present in the page a image descriptor model will be run, t
hese coordianes are gotten from LightOnOCr 
*/
pub struct Page {
    path: String, 
    transcription_type: TranscriptionType,
    img_coordinates: Option<ImageCoordinates> 
}

pub enum TranscriptionType {
    HANDWRITTEN,
    PRINT,
    IMAGE
}

pub struct ImageCoordinates {
    x1: usize,
    x2: usize,
    y1: usize,
    y2: usize,
}