"""Diagnostic reporting and export package."""

from grid_disturbance.reporting.diagnostics import create_structured_report
from grid_disturbance.reporting.export import export_diagnostic_report
from grid_disturbance.reporting.html import generate_html_report, save_html_report
from grid_disturbance.reporting.schemas import (
    ClassificationSection,
    ModelEvidenceSection,
    OscillationCharacteristicsSection,
    SignalEvidenceSection,
    SourceAssessmentSection,
    StructuredDiagnosticReport,
)

__all__ = [
    "ClassificationSection",
    "ModelEvidenceSection",
    "OscillationCharacteristicsSection",
    "SignalEvidenceSection",
    "SourceAssessmentSection",
    "StructuredDiagnosticReport",
    "create_structured_report",
    "export_diagnostic_report",
    "generate_html_report",
    "save_html_report",
]
