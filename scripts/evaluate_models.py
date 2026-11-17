"""Script to evaluate and compare trained models across benchmarks."""

import argparse
from pathlib import Path

import joblib
import numpy as np

from grid_disturbance.data.splitting import split_dataset
from grid_disturbance.ml.evaluation import evaluate_classifier_predictions
from grid_disturbance.synthetic.presets import generate_benchmark_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate disturbance models")
    parser.add_argument("--model-path", type=str, default="models/checkpoints/random_forest.joblib")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    path = Path(args.model_path)
    if not path.is_file():
        print(f"Model file not found: {path}. Train one using scripts/train_models.py first.")
        return

    data = joblib.load(path)
    clf = data["model"]
    vectorizer = data["vectorizer"]
    classes = data["classes"]

    print("Generating evaluation benchmark dataset...")
    signals, df = generate_benchmark_dataset(samples_per_class=args.samples, random_seed=999)
    _, _, test_df, _ = split_dataset(df, target_column="disturbance_type")

    label_map = {c: i for i, c in enumerate(classes)}
    test_sigs = [signals[i].data for i in test_df.index]
    test_lbls = [label_map[test_df.loc[i, "disturbance_type"]] for i in test_df.index]

    x_test = vectorizer.transform(test_sigs)
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)

    report = evaluate_classifier_predictions(
        y_true=np.array(test_lbls),
        y_pred=y_pred,
        y_proba=y_prob,
        class_names=classes,
        model_name=path.stem,
    )

    print(f"\n=== Evaluation Report: {report.model_name} ===")
    print(f"Accuracy: {report.accuracy:.3f}")
    print(f"Macro F1: {report.macro_f1:.3f}")
    print(f"Weighted F1: {report.weighted_f1:.3f}")
    print("\nPer-Class Performance:")
    for c_name, m in report.per_class_metrics.items():
        print(
            f"  {c_name:<22}: Precision = {m.precision:.3f}, Recall = {m.recall:.3f}, F1 = {m.f1_score:.3f} (n={m.support})"
        )


if __name__ == "__main__":
    main()
