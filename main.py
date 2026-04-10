import argparse

from src.interface import process_file, process_file_batch


def main():
    parser = argparse.ArgumentParser(description="Segment-based duration modification tool")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("--output", required=True, help="Output WAV file")

    parser.add_argument("--t1", type=float, help="Segment start time in seconds")
    parser.add_argument("--t2", type=float, help="Segment end time in seconds")
    parser.add_argument("--scale", type=float, help="Time-scaling factor")
    parser.add_argument("--target-duration", type=float, help="Target duration in seconds")

    parser.add_argument("--csv", help="CSV spec file for batch edits")

    args = parser.parse_args()

    single_mode = (
        args.t1 is not None
        or args.t2 is not None
        or args.scale is not None
        or args.target_duration is not None
    )
    batch_mode = args.csv is not None

    if single_mode and batch_mode:
        raise ValueError("Use either single-edit arguments (--t1/--t2 with --scale or --target-duration) or --csv, not both")

    if batch_mode:
        process_file_batch(
            input_path=args.input,
            output_path=args.output,
            csv_path=args.csv,
        )
        return

    if args.t1 is None or args.t2 is None:
        raise ValueError("Single-edit mode requires --t1 and --t2")

    if args.scale is not None and args.target_duration is not None:
        raise ValueError("Provide either --scale or --target-duration, not both")

    if args.scale is None and args.target_duration is None:
        raise ValueError("Single-edit mode requires either --scale or --target-duration")

    process_file(
        input_path=args.input,
        output_path=args.output,
        t1=args.t1,
        t2=args.t2,
        scale=args.scale,
        target_duration=args.target_duration,
    )


if __name__ == "__main__":
    main()