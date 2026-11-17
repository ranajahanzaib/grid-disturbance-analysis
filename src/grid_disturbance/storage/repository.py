"""CRUD repository for saving and retrieving analysis history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grid_disturbance.reporting.schemas import StructuredDiagnosticReport
from grid_disturbance.storage.database import get_db_connection


class AnalysisRepository:
    """Repository handling persistence of disturbance diagnostic reports in SQLite."""

    def __init__(self, db_path: str | Path = "data/storage.db"):
        self.db_path = Path(db_path)

    def save_analysis(self, report: StructuredDiagnosticReport) -> int:
        """Insert analysis report into database and return its row ID."""
        conn = get_db_connection(self.db_path)
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO analyses (
                    event_id, disturbance_type, confidence, dominant_frequency,
                    damping_ratio, is_sustained, candidate_source, summary, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.event_id,
                    report.event_classification.predicted_class,
                    report.event_classification.confidence,
                    report.oscillation_characteristics.dominant_frequency_hz,
                    report.oscillation_characteristics.damping_ratio,
                    1 if report.oscillation_characteristics.is_sustained else 0,
                    report.source_assessment.candidate_origin_channel,
                    report.plain_language_summary,
                    report.model_dump_json(),
                ),
            )
            return int(cursor.lastrowid or 0)

    def list_analyses(self, limit: int = 50) -> list[dict[str, Any]]:
        """List past analyses ordered by most recent first."""
        conn = get_db_connection(self.db_path)
        with conn:
            rows = conn.execute(
                """
                SELECT id, event_id, created_at, disturbance_type, confidence,
                       dominant_frequency, damping_ratio, is_sustained, candidate_source
                FROM analyses ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_analysis_by_id(self, analysis_id: int) -> StructuredDiagnosticReport | None:
        """Fetch full StructuredDiagnosticReport by row ID."""
        conn = get_db_connection(self.db_path)
        with conn:
            row = conn.execute(
                "SELECT report_json FROM analyses WHERE id = ?",
                (analysis_id,),
            ).fetchone()
            if row is None:
                return None
            return StructuredDiagnosticReport.model_validate_json(row["report_json"])

    def get_analysis_by_event_id(self, event_id: str) -> StructuredDiagnosticReport | None:
        """Fetch latest StructuredDiagnosticReport by event ID."""
        conn = get_db_connection(self.db_path)
        with conn:
            row = conn.execute(
                "SELECT report_json FROM analyses WHERE event_id = ? ORDER BY id DESC LIMIT 1",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            return StructuredDiagnosticReport.model_validate_json(row["report_json"])
