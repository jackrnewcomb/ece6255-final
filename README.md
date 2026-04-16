# ECE6255 Term Project: Arbitrary Segment Duration Modification for Speech

## Overview

This project implements a modular framework for arbitrary duration modification of user-specified regions in a speech waveform. The target challenge is **Challenge 3: Arbitrary Modification of Speech Characteristics in Segmental Durations**, which calls for changing the duration of selected portions of a speech signal while preserving essential properties such as pitch contour and spectral characteristics. 

The current repository provides the end-to-end processing pipeline for:
- loading mono WAV audio,
- selecting a segment by time,
- applying a time-scaling algorithm to that segment,
- stitching the modified segment back into the original signal,
- processing either a single edit or a batch of edits from CSV,
- running automated tests,
- and generating basic evaluation plots and metrics.

## Project Scope

The assignment requires code, usage instructions, examples, evaluation, and documentation that can be useful to future students as well as the current course staff.

This repository currently focuses on:
- **single-region duration modification** via either a scale factor or a target duration,
- **batch region modification** via a CSV specification file,
- **evaluation tooling** for waveform/spectrogram comparison and simple metrics,
- and **test coverage** for the framework around the duration-modification algorithm.

## Repository Structure

```text
.
├── main.py                  # Command-line entry point
├── examples/
│   ├── tsignal.wav          # Example input audio
│   ├── sample_input.csv     # Example batch edit specification
│   ├── out.wav
│   └── output.wav
├── scripts/
│   └── evaluate.py          # Evaluation plots and metrics
├── src/
│   ├── audio_io.py          # WAV loading/saving
│   ├── interface.py         # High-level processing pipeline
│   ├── psola.py             # Time-scaling algorithm hook (currently placeholder)
│   ├── segment.py           # Segment extraction utilities
│   ├── stitch.py            # Segment stitching logic
│   └── utils.py             # CSV edit-spec parsing
└── tests/
    ├── conftest.py
    ├── test_interface.py
    ├── test_segment.py
    ├── test_stitch.py
    └── test_utils.py
```

## Requirements

Recommended environment: 
- Python 3.12+
- `numpy`
- `scipy`
- `matplotlib`
- `pytest`

Install dependencies with:

```bash
pip install numpy scipy matplotlib pytest
```

## How the Pipeline Works

The high-level processing flow is:

1. Load a mono WAV file
2. Convert user-specified times t1, t2 into sample indices
3. Split the signal into:
prefix
target segment
suffix
4. Pass the target segment to time_scale_psola(...)
5. Validate the returned processed segment
6. Stitch the output back together
7. Save the modified WAV file

The main algorithm boundary is:

```py
time_scale_psola(segment, sample_rate, scale)
```

This function is expected to return a 1-D processed audio segment whose duration reflects the requested scale factor.

## Running the Tool
### Single Edit Mode

Use single-edit mode when modifying one region of the input file.

**Option A: Specify a scale factor**

```py
python main.py examples/tsignal.wav --output examples/output.wav --t1 0.20 --t2 0.40 --scale 1.5
```

This stretches the region from 0.20 s to 0.40 s by a factor of 1.5.

**Option B: Specify a target duration**

```py
python main.py examples/tsignal.wav --output examples/output.wav --t1 0.20 --t2 0.40 --target-duration 0.50
```

This changes the selected region so that its new duration is 0.50 s.

### Batch Edit Mode

Use batch-edit mode to apply multiple edits from a CSV specification file:

```py
python main.py examples/tsignal.wav --output examples/output.wav --csv examples/sample_input.csv
```

## CSV Format

The CSV file must include:

- `t1`
- `t2`

and at least one of:
- `scale`
- `target_duration`

Each row must specify exactly one of scale or target_duration.

Example:

```csv
t1,t2,scale,target_duration
0.20,0.40,1.5,
0.60,0.80,,0.30
```

### Batch Timing Convention

In batch mode, edit times are interpreted relative to the original input file timeline, not the progressively modified output timeline.

Therefore, if an earlier edit changes duration,
later edit times are automatically remapped onto the current modified signal so they still refer to the intended original regions.

### Batch Edit Restrictions

Overlapping edits in the original timeline are not supported and will raise an error.

For example, the following is invalid:

```csv
t1,t2,scale
0.20,0.50,1.2
0.40,0.70,0.8
```

because the second region overlaps the first.

## Evaluation

The repository includes an evaluation script for comparing original and modified audio.

### Basic usage
```py
python scripts/evaluate.py --original examples/tsignal.wav --modified examples/output.wav --report-dir reports/basic_eval
```

This generates:

- waveform comparison plot
- spectrogram of original
- spectrogram of modified
- basic metrics text file

### Region-focused evaluation

```py
python scripts/evaluate.py --original examples/tsignal.wav --modified examples/output.wav --report-dir reports/region_eval --t1 0.20 --t2 0.40 --margin 0.10
```

This additionally generates:

- waveform plot focused on the edited region
- region spectrograms
- region-specific metrics

## Testing

Run the test suite with:

```py
pytest
```

The tests currently cover:

- segment extraction behavior
- stitching behavior
- CSV parsing
- single-edit processing
- batch-edit processing
- target-duration handling
- batch timeline remapping after duration-changing edits
- rejection of overlapping batch edits
