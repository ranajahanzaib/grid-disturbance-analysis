"""End-to-end integration test for the complete analytical pipeline."""

from grid_disturbance.analysis.diagnosis import DisturbanceDiagnosticEngine
from grid_disturbance.data.preprocessing import SignalPreprocessor
from grid_disturbance.data.validation import validate_signal_array
from grid_disturbance.reporting.diagnostics import create_structured_report
from grid_disturbance.reporting.export import export_diagnostic_report
from grid_disturbance.signals.features import extract_signal_features
from grid_disturbance.storage.repository import AnalysisRepository
from grid_disturbance.synthetic.generator import SyntheticSignalGenerator


def test_full_analytical_pipeline_integration(tmp_path):
    # 1. Input Signal
    gen = SyntheticSignalGenerator(random_seed=42)
    sig = gen.generate_forced_oscillation(
        frequency=1.10, amplitude=1.2, sampling_rate=30.0, duration_sec=15.0
    )

    # 2. Validation
    val = validate_signal_array(sig.data, sampling_rate=30.0)
    assert val.is_valid

    # 3. Preprocessing
    preprocessor = SignalPreprocessor()
    clean_sig, record = preprocessor.process(sig.data, sampling_rate=30.0)
    assert len(clean_sig) == len(sig.data)

    # 4. Feature Extraction
    feats = extract_signal_features(clean_sig, sampling_rate=30.0)
    assert len(feats) > 20

    # 5. Disturbance Detection & Diagnosis
    engine = DisturbanceDiagnosticEngine(preprocessor=preprocessor)
    diag = engine.diagnose_signal(sig.data, sampling_rate=30.0, event_id="int_test_001")
    assert diag.disturbance_type == "forced_oscillation"
    assert diag.confidence >= 0.60

    # 6. Structured Report Generation
    report = create_structured_report(diag)
    assert report.event_id == "int_test_001"
    assert report.event_classification.predicted_class == "forced_oscillation"

    # 7. Export to JSON and HTML
    export_dir = tmp_path / "reports"
    exported = export_diagnostic_report(report, output_dir=export_dir, formats=["json", "html"])
    assert "json" in exported and exported["json"].is_file()
    assert "html" in exported and exported["html"].is_file()

    # 8. Local SQLite Storage Persistence
    db_file = tmp_path / "test_storage.db"
    repo = AnalysisRepository(db_path=db_file)
    row_id = repo.save_analysis(report)
    assert row_id >= 1

    loaded = repo.get_analysis_by_id(row_id)
    assert loaded is not None
    assert loaded.event_id == "int_test_001"
    assert loaded.event_classification.predicted_class == "forced_oscillation"
