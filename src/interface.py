from src.audio_io import load_wav, save_wav
from src.segment import extract_segments
from src.psola import time_scale_psola
from src.stitch import stitch_segments

def process_file(input_path: str, output_path: str, t1: float, t2: float, scale: float):
    sample_rate, samples = load_wav(input_path)

    prefix, segment, suffix = extract_segments(samples, sample_rate, t1, t2)
    processed_segment = time_scale_psola(segment, sample_rate, scale)
    output = stitch_segments(prefix, processed_segment, suffix)

    save_wav(output_path, sample_rate, output)