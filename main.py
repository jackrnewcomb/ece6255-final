import argparse
from src.interface import process_file

def main():
    parser = argparse.ArgumentParser(description="Segment-based duration modification tool")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("--t1", type=float, required=True, help="Segment start time in seconds")
    parser.add_argument("--t2", type=float, required=True, help="Segment end time in seconds")
    parser.add_argument("--scale", type=float, required=True, help="Time-scaling factor")
    parser.add_argument("--output", required=True, help="Output WAV file")

    args = parser.parse_args()

    process_file(
        input_path=args.input,
        output_path=args.output,
        t1=args.t1,
        t2=args.t2,
        scale=args.scale,
    )

if __name__ == "__main__":
    main()