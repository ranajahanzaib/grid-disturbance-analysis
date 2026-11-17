"""Standalone HTML diagnostic report generator."""

from __future__ import annotations

from pathlib import Path

from grid_disturbance.reporting.schemas import StructuredDiagnosticReport


def generate_html_report(report: StructuredDiagnosticReport) -> str:
    """Generate self-contained, publication-ready HTML report with modern CSS styling."""
    cls_sec = report.event_classification
    osc = report.oscillation_characteristics
    src = report.source_assessment

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Disturbance Diagnostic Report - {report.event_id}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: #1e293b;
      background: #f8fafc;
      padding: 30px;
      margin: 0;
    }}
    .container {{
      max-width: 1000px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      padding: 36px;
    }}
    header {{
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 18px;
      margin-bottom: 24px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 9999px;
      font-weight: 600;
      font-size: 0.85rem;
      text-transform: uppercase;
      background: #e0f2fe;
      color: #0369a1;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .card {{
      background: #f1f5f9;
      border-radius: 8px;
      padding: 18px;
      border-left: 4px solid #3b82f6;
    }}
    .card h3 {{
      margin-top: 0;
      font-size: 1.05rem;
      color: #0f172a;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
    }}
    th, td {{
      text-align: left;
      padding: 10px 14px;
      border-bottom: 1px solid #e2e8f0;
    }}
    th {{
      background: #f8fafc;
      color: #475569;
      font-size: 0.85rem;
    }}
    .alert-box {{
      background: #fffbeb;
      border: 1px solid #fde68a;
      border-left: 4px solid #f59e0b;
      padding: 14px;
      border-radius: 6px;
      margin: 16px 0;
      color: #92400e;
    }}
    ul {{
      padding-left: 20px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <h1 style="margin:0 0 6px 0; color:#0f172a;">Power Grid Disturbance Diagnostic Report</h1>
          <div style="color:#64748b; font-size:0.9rem;">Event ID: <strong>{report.event_id}</strong> | Generated: {report.generated_at}</div>
        </div>
        <span class="badge">{cls_sec.predicted_class}</span>
      </div>
    </header>

    <div class="card" style="margin-bottom: 24px; border-left-color: #10b981;">
      <h3>Executive Summary</h3>
      <p style="margin: 0; font-size: 1.05rem; color: #1e293b;">{report.plain_language_summary}</p>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3>Oscillation Characteristics</h3>
        <table>
          <tr><td>Dominant Frequency</td><td><strong>{osc.dominant_frequency_hz:.2f} Hz</strong></td></tr>
          <tr><td>Damping Ratio (ζ)</td><td><strong>{osc.damping_ratio * 100:.1f}%</strong> ({osc.damping_classification})</td></tr>
          <tr><td>Decay / Growth Rate</td><td>{osc.decay_rate:.3f} s⁻¹</td></tr>
          <tr><td>Frequency Drift</td><td>{osc.frequency_drift_hz:.3f} Hz</td></tr>
          <tr><td>Sustained Behavior</td><td>{"Yes (Forced/Undamped)" if osc.is_sustained else "No (Decaying Ringdown)"}</td></tr>
        </table>
      </div>

      <div class="card" style="border-left-color: #8b5cf6;">
        <h3>Source & Origin Assessment</h3>
        <p><strong>Supported:</strong> {"Yes" if src.is_supported else "No (Single channel)"}</p>
        <p><strong>Candidate Origin:</strong> {src.candidate_origin_region or "Unsupported"}</p>
        <p><strong>Assessment Confidence:</strong> {src.confidence * 100:.0f}%</p>
        <div style="font-size:0.85rem; color:#64748b;">
          {"Identified via cross-spectral phase lead across synchronized PMU channels." if src.is_supported else "Requires minimum 2 synchronized PMU channels for directional inference."}
        </div>
      </div>
    </div>

    <h3>Detailed Engineering Observations</h3>
    <ul>
      {"".join(f"<li>{obs}</li>" for obs in report.technical_observations)}
    </ul>

    <div class="alert-box">
      <strong>Engineering Limitations & Physical Causation Boundary:</strong>
      <ul>
        {"".join(f"<li>{lim}</li>" for lim in report.engineering_limitations)}
      </ul>
    </div>
  </div>
</body>
</html>
"""
    return html


def save_html_report(report: StructuredDiagnosticReport, output_path: str | Path) -> Path:
    """Save HTML report to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_content = generate_html_report(report)
    with path.open("w", encoding="utf-8") as f:
        f.write(html_content)
    return path
