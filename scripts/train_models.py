"""Script to train and persist disturbance classification models."""

import argparse
from pathlib import Path

import joblib
import torch

from grid_disturbance.data.splitting import split_dataset
from grid_disturbance.ml.training import train_classical_model, train_deep_learning_model
from grid_disturbance.synthetic.presets import generate_benchmark_dataset


def main():
    parser = argparse.ArgumentParser(description="Train disturbance classification models")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=[
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "temporal_cnn",
            "spectrogram_cnn",
        ],
    )
    parser.add_argument("--samples", type=int, default=40, help="Samples per class")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for deep learning models")
    parser.add_argument(
        "--output-dir", type=str, default="models/checkpoints", help="Model checkpoint directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating training benchmark data ({args.samples} samples/class)...")
    signals, df = generate_benchmark_dataset(samples_per_class=args.samples, random_seed=args.seed)
    train_df, val_df, test_df, _ = split_dataset(
        df, target_column="disturbance_type", random_seed=args.seed
    )

    classes = sorted(df["disturbance_type"].unique().tolist())
    label_map = {c: i for i, c in enumerate(classes)}

    train_sigs = [signals[i].data for i in train_df.index]
    train_lbls = [label_map[train_df.loc[i, "disturbance_type"]] for i in train_df.index]
    val_sigs = [signals[i].data for i in val_df.index]
    val_lbls = [label_map[val_df.loc[i, "disturbance_type"]] for i in val_df.index]
    test_sigs = [signals[i].data for i in test_df.index]
    test_lbls = [label_map[test_df.loc[i, "disturbance_type"]] for i in test_df.index]

    if args.model in ["temporal_cnn", "spectrogram_cnn"]:
        print(f"Training deep learning model: {args.model} for {args.epochs} epochs...")
        net, report, _ = train_deep_learning_model(
            train_sigs,
            train_lbls,
            val_sigs,
            val_lbls,
            test_sigs,
            test_lbls,
            model_type=args.model,
            num_classes=len(classes),
            epochs=args.epochs,
            class_names=classes,
        )
        save_path = out_dir / f"{args.model}.pt"
        torch.save(net.state_dict(), save_path)
    else:
        print(f"Training classical model: {args.model}...")
        clf, vec, report = train_classical_model(
            train_sigs,
            train_lbls,
            test_sigs,
            test_lbls,
            model_type=args.model,
            class_names=classes,
            random_state=args.seed,
        )
        save_path = out_dir / f"{args.model}.joblib"
        joblib.dump({"model": clf, "vectorizer": vec, "classes": classes}, save_path)

    print(f"Model saved to: {save_path}")
    print(f"Test Accuracy: {report.accuracy:.3f}")
    print(f"Macro F1: {report.macro_f1:.3f}")


if __name__ == "__main__":
    main()
