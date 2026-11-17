"""Command Line Interface (CLI) for Grid Disturbance Analysis."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from grid_disturbance.analysis.diagnosis import DisturbanceDiagnosticEngine
from grid_disturbance.data.gesl import GESLDatasetManager
from grid_disturbance.data.loaders import load_signal_from_file
from grid_disturbance.data.validation import validate_signal_array
from grid_disturbance.reporting.diagnostics import create_structured_report
from grid_disturbance.reporting.export import export_diagnostic_report
from grid_disturbance.signals.features import extract_signal_features
from grid_disturbance.signals.spectral import analyze_spectrum
from grid_disturbance.storage.repository import AnalysisRepository
from grid_disturbance.synthetic.presets import (
    generate_benchmark_dataset,
)

app = typer.Typer(
    name="grid-disturbance",
    help="Power Grid Disturbance Analysis & Source Localization Engine",
    add_completion=False,
)
data_app = typer.Typer(name="data", help="Dataset discovery, preparation, and validation")
signal_app = typer.Typer(name="signal", help="Signal processing, spectral analysis, and features")
models_app = typer.Typer(name="models", help="Model management and inspection")

app.add_typer(data_app)
app.add_typer(signal_app)
app.add_typer(models_app)

console = Console()


# -------------------------------------------------------------
# Data Subcommands
# -------------------------------------------------------------
@data_app.command("inspect")
def data_inspect(
    file_path: str = typer.Argument(None, help="Path to dataset CSV to inspect"),
):
    """Inspect dataset schema, missing values, duplicates, and quality."""
    manager = GESLDatasetManager()
    report = manager.inspect_dataset(file_path)

    table = Table(title=f"Dataset Inspection: {report.dataset_name}")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Valid", str(report.is_valid))
    table.add_row("Total Records", f"{report.total_records:,}")
    table.add_row("Total Events", f"{report.total_events:,}")
    table.add_row("Sampling Rate", f"{report.sampling_rate} Hz")
    table.add_row("Missing Values", f"{report.missing_value_percentage:.2f}%")
    table.add_row("Duplicate Records", str(report.duplicate_records_count))
    table.add_row("Event Grouping", str(report.has_event_grouping))

    console.print(table)

    if report.class_distribution:
        dist_table = Table(title="Class Distribution")
        dist_table.add_column("Disturbance Class", style="green")
        dist_table.add_column("Count", style="white")
        for cls_name, count in report.class_distribution.items():
            dist_table.add_row(cls_name, f"{count:,}")
        console.print(dist_table)

    if report.quality_notes:
        console.print(
            Panel(
                "\n".join(f"• {note}" for note in report.quality_notes),
                title="Quality Notes",
                border_style="yellow",
            )
        )


@data_app.command("prepare")
def data_prepare(
    samples_per_class: int = typer.Option(
        50, "--samples", "-n", help="Samples per disturbance class"
    ),
    output: str = typer.Option(
        "gesl_synthetic_benchmark.csv", "--output", "-o", help="Output filename"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducibility"),
):
    """Prepare and register a reproducible benchmark dataset."""
    with console.status("[bold green]Generating benchmark dataset..."):
        manager = GESLDatasetManager()
        out_path, manifest = manager.prepare_benchmark_dataset(
            samples_per_class=samples_per_class,
            output_name=output,
            random_seed=seed,
        )

    console.print(
        f"[bold green]✔ Successfully generated benchmark dataset:[/bold green] {out_path}"
    )
    console.print(
        f"Total Records: {manifest.total_records} (Train: {manifest.partitions['train'].num_records}, Val: {manifest.partitions['validation'].num_records}, Test: {manifest.partitions['test'].num_records})"
    )
    console.print(f"Manifest Checksum SHA256: [dim]{manifest.checksum_sha256}[/dim]")


@data_app.command("validate")
def data_validate(
    file_path: str = typer.Argument(..., help="Path to signal or dataset CSV to validate"),
):
    """Validate signal array or dataset for sampling consistency and defects."""
    try:
        t, data, sr, channels = load_signal_from_file(file_path)
        sig = data if data.ndim == 1 else data[0, :]
        report = validate_signal_array(sig, sampling_rate=sr)

        if report.is_valid:
            console.print(
                f"[bold green]✔ Signal is valid![/bold green] ({report.num_samples} samples @ {sr} Hz, duration {report.duration_sec:.2f}s, {len(channels)} channels)"
            )
        else:
            console.print(f"[bold red]✘ Signal validation failed:[/bold red] {report.errors}")

        if report.warnings:
            console.print(f"[yellow]Warnings: {report.warnings}[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error loading file:[/bold red] {e}")


@data_app.command("stats")
def data_stats():
    """Display statistics for all datasets in data directory."""
    manager = GESLDatasetManager()
    manifest_files = list(manager.manifest_dir.glob("*.json"))

    if not manifest_files:
        console.print(
            "[yellow]No dataset manifests found in data/manifests. Run 'grid-disturbance data prepare' first.[/yellow]"
        )
        return

    table = Table(title="Registered Datasets & Manifests")
    table.add_column("Dataset Name", style="cyan")
    table.add_column("Source", style="green")
    table.add_column("Records", style="white")
    table.add_column("Split Strategy", style="magenta")
    table.add_column("Created", style="dim")

    for mf in manifest_files:
        with mf.open("r", encoding="utf-8") as f:
            data = json.load(f)
            table.add_row(
                data.get("dataset_name", mf.stem),
                data.get("source", "unknown"),
                f"{data.get('total_records', 0):,}",
                data.get("split_strategy", "n/a"),
                data.get("created_at", "n/a")[:10],
            )
    console.print(table)


# -------------------------------------------------------------
# Signal Subcommands
# -------------------------------------------------------------
@signal_app.command("analyze")
def signal_analyze(
    file_path: str = typer.Argument(..., help="Path to signal file (.csv, .npy)"),
    sampling_rate: float = typer.Option(30.0, "--sr", help="Sampling rate in Hz if not in file"),
):
    """Analyze signal spectral peaks, damping, and forced/natural behavior."""
    t, data, sr, channels = load_signal_from_file(file_path, sampling_rate=sampling_rate)
    sig = data if data.ndim == 1 else data[0, :]

    spec = analyze_spectrum(sig, sampling_rate=sr)
    engine = DisturbanceDiagnosticEngine()
    diag = engine.diagnose_signal(data, sampling_rate=sr, channel_names=channels)

    table = Table(title=f"Signal Analysis: {Path(file_path).name}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Measured Value", style="white")

    table.add_row("Dominant Frequency", f"{spec.dominant_frequency:.2f} Hz")
    table.add_row("Peak Prominence", f"{spec.peaks[0].prominence:.3f}" if spec.peaks else "0.0")
    table.add_row("Spectral Concentration", f"{spec.spectral_concentration:.2f}")
    table.add_row("Spectral Entropy", f"{spec.spectral_entropy:.2f}")
    table.add_row(
        "Estimated Damping Ratio (ζ)", f"{diag.oscillations.primary_damping_ratio * 100:.1f}%"
    )
    table.add_row(
        "Disturbance Verdict",
        f"[bold green]{diag.disturbance_type}[/bold green] (confidence {diag.confidence * 100:.0f}%)",
    )

    console.print(table)
    console.print(Panel(diag.summary, title="Summary Verdict", border_style="blue"))


@signal_app.command("features")
def signal_features(
    file_path: str = typer.Argument(..., help="Path to signal file"),
    sampling_rate: float = typer.Option(30.0, "--sr", help="Sampling rate"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Extract full time-domain, frequency-domain, and oscillation features."""
    t, data, sr, channels = load_signal_from_file(file_path, sampling_rate=sampling_rate)
    sig = data if data.ndim == 1 else data[0, :]

    feats = extract_signal_features(sig, sampling_rate=sr)

    if json_output:
        console.print_json(data=feats)
    else:
        table = Table(title=f"Extracted Features ({len(feats)} total)")
        table.add_column("Feature Name", style="cyan")
        table.add_column("Value", style="magenta")
        for k, v in sorted(feats.items()):
            table.add_row(k, f"{v:.4f}")
        console.print(table)


# -------------------------------------------------------------
# Train, Evaluate, Predict, Cluster
# -------------------------------------------------------------
@app.command("train")
def train_command(
    model: str = typer.Option(
        "random_forest",
        "--model",
        "-m",
        help="Model: random_forest, extra_trees, gradient_boosting, temporal_cnn, spectrogram_cnn",
    ),
    samples_per_class: int = typer.Option(40, "--samples", "-n", help="Training samples per class"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Epochs for deep learning models"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
):
    """Train a classical ML model or deep neural network on disturbance signals."""
    from grid_disturbance.data.splitting import split_dataset
    from grid_disturbance.ml.training import train_classical_model, train_deep_learning_model

    console.print(f"[bold blue]Initiating training pipeline for model:[/bold blue] {model}")

    with console.status("[bold green]Generating training benchmark signals..."):
        signals, df = generate_benchmark_dataset(
            samples_per_class=samples_per_class, random_seed=seed
        )
        train_df, val_df, test_df, _ = split_dataset(
            df, target_column="disturbance_type", random_seed=seed
        )

        classes = sorted(df["disturbance_type"].unique().tolist())
        label_map = {c: i for i, c in enumerate(classes)}

        train_sigs = [signals[i].data for i in train_df.index]
        train_lbls = [label_map[train_df.loc[i, "disturbance_type"]] for i in train_df.index]

        val_sigs = [signals[i].data for i in val_df.index]
        val_lbls = [label_map[val_df.loc[i, "disturbance_type"]] for i in val_df.index]

        test_sigs = [signals[i].data for i in test_df.index]
        test_lbls = [label_map[test_df.loc[i, "disturbance_type"]] for i in test_df.index]

    if model in ["temporal_cnn", "spectrogram_cnn"]:
        with console.status(f"[bold green]Training {model} for {epochs} epochs..."):
            net, report, _ = train_deep_learning_model(
                train_sigs,
                train_lbls,
                val_sigs,
                val_lbls,
                test_sigs,
                test_lbls,
                model_type=model,
                num_classes=len(classes),
                epochs=epochs,
                class_names=classes,
            )
    else:
        with console.status(f"[bold green]Training {model}..."):
            clf, vec, report = train_classical_model(
                train_sigs,
                train_lbls,
                test_sigs,
                test_lbls,
                model_type=model,
                class_names=classes,
                random_state=seed,
            )

    console.print("[bold green]✔ Training Completed![/bold green]")
    table = Table(title=f"Evaluation Metrics: {report.model_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="white")
    table.add_row("Accuracy", f"{report.accuracy:.3f}")
    table.add_row("Macro Precision", f"{report.macro_precision:.3f}")
    table.add_row("Macro Recall", f"{report.macro_recall:.3f}")
    table.add_row("Macro F1", f"{report.macro_f1:.3f}")
    table.add_row("Weighted F1", f"{report.weighted_f1:.3f}")
    console.print(table)


@app.command("predict")
def predict_command(
    file_path: str = typer.Argument(..., help="Path to signal file to classify"),
    sampling_rate: float = typer.Option(30.0, "--sr", help="Sampling rate"),
):
    """Run full disturbance diagnosis and classification on a signal."""
    t, data, sr, channels = load_signal_from_file(file_path, sampling_rate=sampling_rate)
    engine = DisturbanceDiagnosticEngine()
    diag = engine.diagnose_signal(data, sampling_rate=sr, channel_names=channels)

    console.print(f"[bold cyan]File:[/bold cyan] {file_path}")
    console.print(
        f"[bold green]Predicted Disturbance:[/bold green] {diag.disturbance_type} (Confidence: {diag.confidence * 100:.1f}%)"
    )
    console.print(f"[bold]Dominant Frequency:[/bold] {diag.spectrum.dominant_frequency:.2f} Hz")
    console.print(
        f"[bold]Estimated Damping:[/bold] {diag.oscillations.primary_damping_ratio * 100:.1f}%"
    )
    console.print(Panel(diag.summary, title="Diagnostic Summary"))


@app.command("report")
def report_command(
    file_path: str = typer.Argument(..., help="Path to signal file"),
    format_type: str = typer.Option(
        "html", "--format", "-f", help="Output format: html, json, or csv"
    ),
    output_dir: str = typer.Option("reports", "--output-dir", "-o", help="Output directory"),
    save_db: bool = typer.Option(
        True, "--save-db/--no-db", help="Persist to local SQLite database"
    ),
):
    """Generate and export a structured technical diagnostic report."""
    t, data, sr, channels = load_signal_from_file(file_path)
    engine = DisturbanceDiagnosticEngine()
    diag = engine.diagnose_signal(
        data, sampling_rate=sr, channel_names=channels, event_id=Path(file_path).stem
    )
    report = create_structured_report(diag)

    exported = export_diagnostic_report(report, output_dir=output_dir, formats=[format_type])

    if save_db:
        repo = AnalysisRepository()
        row_id = repo.save_analysis(report)
        console.print(f"[dim]Persisted analysis to SQLite database (id={row_id})[/dim]")

    console.print(
        f"[bold green]✔ Diagnostic report generated:[/bold green] {exported.get(format_type, list(exported.values())[0])}"
    )


@app.command("cluster")
def cluster_command(
    samples_per_class: int = typer.Option(30, "--samples", "-n", help="Samples per class"),
    k: int = typer.Option(5, "--k", help="Number of clusters"),
):
    """Perform unsupervised event clustering on disturbance signatures."""
    from grid_disturbance.clustering.algorithms import cluster_disturbance_features
    from grid_disturbance.ml.features import SignalFeatureVectorizer

    with console.status("[bold green]Generating signals and extracting feature vectors..."):
        signals, df = generate_benchmark_dataset(samples_per_class=samples_per_class)
        vec = SignalFeatureVectorizer(sampling_rate=30.0)
        x_mat = vec.fit_transform([s.data for s in signals])
        res = cluster_disturbance_features(x_mat, algorithm="kmeans", n_clusters=k)

    console.print("[bold green]✔ Unsupervised Clustering Completed[/bold green]")
    table = Table(title="Cluster Discovery Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Total Samples", str(len(signals)))
    table.add_row("Clusters Discovered", str(res.n_clusters))
    table.add_row("Silhouette Score", f"{res.silhouette:.3f}")
    table.add_row("Davies-Bouldin Index", f"{res.davies_bouldin:.3f}")
    table.add_row("Anomalies Detected (Isolation Forest)", str(len(res.anomaly_indices)))
    console.print(table)


@models_app.command("list")
def models_list():
    """List supported classical and deep learning model architectures."""
    table = Table(title="Supported Disturbance Classification Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Inputs", style="magenta")
    table.add_column("Description", style="white")

    table.add_row(
        "random_forest",
        "Classical Ensemble",
        "Tabular Features",
        "Balanced multi-tree ensemble; fast and robust",
    )
    table.add_row(
        "extra_trees",
        "Classical Ensemble",
        "Tabular Features",
        "Extremely randomized trees with lower variance",
    )
    table.add_row(
        "gradient_boosting",
        "Classical Boosting",
        "Tabular Features",
        "Gradient boosted decision trees",
    )
    table.add_row(
        "logistic_regression",
        "Classical Linear",
        "Standardized Features",
        "L2-regularized linear baseline with balanced weights",
    )
    table.add_row(
        "temporal_cnn",
        "Deep Learning",
        "1D Signal Windows",
        "1D Temporal CNN with multi-scale Conv1D blocks",
    )
    table.add_row(
        "spectrogram_cnn",
        "Deep Learning",
        "2D Spectrograms",
        "2D CNN extracting time-frequency patterns",
    )

    console.print(table)


@app.command("app")
def app_command(
    port: int = typer.Option(8501, "--port", "-p", help="Port for Streamlit app"),
):
    """Launch the interactive Streamlit engineering diagnostic application."""
    app_path = Path(__file__).parent / "app" / "main.py"
    console.print(f"[bold green]Launching Streamlit application on port {port}...[/bold green]")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    subprocess.run(cmd)


if __name__ == "__main__":
    app()
