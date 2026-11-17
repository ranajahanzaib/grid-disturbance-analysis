"""Exporting diagnostic results and features to JSON, CSV, and HTML."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from grid_disturbance.reporting.html import save_html_report
from grid_disturbance.reporting.schemas import StructuredDiagnosticReport


def export_diagnostic_report(
    report: StructuredDiagnosticReport,
    output_dir: str | Path = "reports",
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """Export diagnostic report to specified formats (json, html, csv)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if formats is None:
        formats = ["json", "html"]

    base_name = f"report_{report.event_id}"
    exported: dict[str, Path] = {}

    if "json" in formats:
        json_path = out_dir / f"{base_name}.json"
        with json_path.open("w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        exported["json"] = json_path

    if "html" in formats:
        html_path = out_dir / f"{base_name}.html"
        save_html_report(report, html_path)
        exported["html"] = html_path

    if "csv" in formats:
        csv_path = out_dir / f"{base_name}_features.csv"
        # Export derived features flat table
        flat_data = {
            "event_id": report.event_id,
            "predicted_class": report.event_classification.predicted_class,
            "confidence": report.event_classification.confidence,
            **report.derived_features,
        }
        df = pd.DataFrame([flat_data])
        df.to_csv(csv_path, index=False)
        exported["csv"] = csv_path

    return exported
