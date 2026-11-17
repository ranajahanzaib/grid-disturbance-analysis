"""Script to prepare and validate benchmark grid disturbance datasets."""

import argparse

from grid_disturbance.data.gesl import GESLDatasetManager


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark dataset")
    parser.add_argument("--samples", type=int, default=50, help="Samples per disturbance class")
    parser.add_argument(
        "--output", type=str, default="gesl_synthetic_benchmark.csv", help="Output filename"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Preparing dataset with {args.samples} samples/class...")
    mgr = GESLDatasetManager()
    out_path, manifest = mgr.prepare_benchmark_dataset(
        samples_per_class=args.samples,
        output_name=args.output,
        random_seed=args.seed,
    )
    print(f"Dataset successfully created at: {out_path}")
    print(f"Manifest written to: data/manifests/{out_path.stem}_manifest.json")
    print(f"Total records: {manifest.total_records}")


if __name__ == "__main__":
    main()
