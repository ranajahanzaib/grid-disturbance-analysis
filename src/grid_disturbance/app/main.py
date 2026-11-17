"""Interactive Engineering Diagnostic Application for Power Grid Disturbances."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Grid Disturbance Analysis Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from grid_disturbance.analysis.diagnosis import DisturbanceDiagnosticEngine
from grid_disturbance.app.components.plots import (
    plot_clustering_scatter,
    plot_spectrogram,
    plot_spectrum_and_psd,
    plot_time_series,
)
from grid_disturbance.clustering.algorithms import cluster_disturbance_features
from grid_disturbance.data.gesl import GESLDatasetManager
from grid_disturbance.data.preprocessing import SignalPreprocessor
from grid_disturbance.data.validation import validate_signal_array
from grid_disturbance.ml.features import SignalFeatureVectorizer
from grid_disturbance.reporting.diagnostics import create_structured_report
from grid_disturbance.signals.envelopes import extract_hilbert_envelope
from grid_disturbance.signals.spectral import analyze_spectrum
from grid_disturbance.signals.transforms import compute_stft
from grid_disturbance.storage.repository import AnalysisRepository
from grid_disturbance.synthetic.presets import (
    generate_benchmark_dataset,
    generate_single_test_signal,
)
from grid_disturbance.synthetic.scenarios import (
    generate_multichannel_forced_oscillation_scenario,
    generate_two_area_inter_area_scenario,
)

# -------------------------------------------------------------
# App Header & Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.title("⚡ Grid Disturbance")
st.sidebar.caption("Power System Oscillation Analysis & Source Localization")

section = st.sidebar.radio(
    "Navigation",
    [
        "1. Overview & Statistics",
        "2. Signal Explorer",
        "3. Signal Analysis",
        "4. Disturbance Diagnosis",
        "5. Event Comparison",
        "6. Event Clustering",
        "7. Model Laboratory",
        "8. Analysis History",
    ],
)

repo = AnalysisRepository()
diagnostic_engine = DisturbanceDiagnosticEngine()
preprocessor = SignalPreprocessor()

# -------------------------------------------------------------
# Section 1: Overview & Statistics
# -------------------------------------------------------------
if section == "1. Overview & Statistics":
    st.title("Power Grid Disturbance Analysis Engine")
    st.markdown(
        """
        An open-source scientific computing and signal-processing platform designed to
        characterize, classify, and isolate the origin of oscillatory disturbances in power systems.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Supported Disturbance Types", "6 Classes")
    col2.metric("PMU Sampling Standards", "30 / 60 Hz")
    col3.metric("Analysis Engine", "Physics-Signal Hybrid")
    col4.metric("Dataset Integration", "GESL & Synthetic")

    st.subheader("Disturbance Taxonomy")
    tax_col1, tax_col2 = st.columns(2)
    with tax_col1:
        st.markdown(
            r"""
            * **Forced Oscillations:** Sustained periodic disturbances driven by malfunctioning equipment, cyclic loads, or control interactions.
            * **Natural Ring-Downs:** Transient oscillatory responses following faults or line tripping; characterized by positive damping ($\zeta \ge 5\%$).
            * **Poorly Damped Modes:** Inter-area or plant oscillations with low damping ($0 < \zeta < 3\%$), prone to resonance.
            """
        )
    with tax_col2:
        st.markdown(
            """
            * **Frequency Excursions:** Large-scale active power imbalances leading to high RoCoF ($df/dt$) and frequency nadir/zenith.
            * **Overlapping Oscillations:** Simultaneous multi-mode events (e.g., concurrent 0.3 Hz inter-area and 1.2 Hz forced oscillation).
            * **Baseline Ambient Noise:** Normal power grid fluctuations and colored measurement noise.
            """
        )

    st.subheader("Dataset Provenance & Real-World Characteristics")
    st.info(
        """
        **Primary Data Source:** Grid Event Signature Library (GESL) developed by ORNL / LLNL / PNNL.
        - **Total Records:** ~203,400 records (~167,400 forced oscillations, ~36,000 ring-downs).
        - **Integrity Boundary:** The application strictly separates real measured grid statistics from synthetic benchmark generators.
        """
    )

    manager = GESLDatasetManager()
    rep = manager.inspect_dataset()
    st.write("Current Registered Dataset Status:", rep.model_dump())


# -------------------------------------------------------------
# Section 2: Signal Explorer
# -------------------------------------------------------------
elif section == "2. Signal Explorer":
    st.title("Signal Explorer & Data Ingestion")

    source_type = st.radio(
        "Select Signal Source:",
        ["Load Synthetic Preset / Scenario", "Upload Custom CSV File"],
        horizontal=True,
    )

    if source_type == "Load Synthetic Preset / Scenario":
        preset_choice = st.selectbox(
            "Choose Disturbance Scenario:",
            [
                "Forced Oscillation (0.85 Hz, Sustained)",
                "Damped Ring-Down (0.65 Hz, 8% Damping)",
                "Poorly Damped Oscillation (0.45 Hz, 1.5% Damping)",
                "Frequency Excursion (Generation Loss, RoCoF = -0.30 Hz/s)",
                "Overlapping Modes (0.35 Hz + 1.25 Hz)",
                "4-Bus Multi-Channel Network (Forced Oscillation Source at Bus 1)",
                "2-Area Multi-Channel Network (Inter-Area 0.40 Hz)",
            ],
        )

        if "4-Bus" in preset_choice:
            scenario = generate_multichannel_forced_oscillation_scenario()
            t, sig_data, sr, channels = (
                scenario.time,
                scenario.signals,
                scenario.sampling_rate,
                scenario.channel_names,
            )
        elif "2-Area" in preset_choice:
            scenario = generate_two_area_inter_area_scenario()
            t, sig_data, sr, channels = (
                scenario.time,
                scenario.signals,
                scenario.sampling_rate,
                scenario.channel_names,
            )
        else:
            cls_key = (
                "forced_oscillation"
                if "Forced" in preset_choice
                else "damped_ringdown"
                if "Damped Ring-Down" in preset_choice
                else "poorly_damped"
                if "Poorly Damped" in preset_choice
                else "frequency_excursion"
                if "Frequency Excursion" in preset_choice
                else "overlapping_events"
            )
            syn_sig = generate_single_test_signal(cls_key)
            t, sig_data, sr, channels = (
                syn_sig.time,
                syn_sig.data,
                syn_sig.ground_truth.sampling_rate,
                ["channel_1"],
            )

    else:
        uploaded_file = st.file_uploader(
            "Upload Synchrophasor CSV (must contain time and signal columns)", type=["csv"]
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10))
            time_col = st.selectbox("Select Time Column", ["None"] + df.columns.tolist())
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if time_col != "None" and time_col in numeric_cols:
                numeric_cols.remove(time_col)
            selected_ch = st.multiselect(
                "Select Measurement Channels", numeric_cols, default=numeric_cols[:1]
            )
            sr = st.number_input("Sampling Rate (Hz)", min_value=1.0, max_value=200.0, value=30.0)

            if selected_ch:
                channels = selected_ch
                if len(selected_ch) == 1:
                    sig_data = df[selected_ch[0]].to_numpy(dtype=np.float64)
                else:
                    sig_data = df[selected_ch].to_numpy(dtype=np.float64).T
                t = df[time_col].to_numpy() if time_col != "None" else np.arange(len(df)) / sr
            else:
                st.stop()
        else:
            st.info("Upload a CSV file or switch to 'Load Synthetic Preset' above.")
            st.stop()

    st.session_state["current_signal"] = sig_data
    st.session_state["current_time"] = t
    st.session_state["current_sr"] = sr
    st.session_state["current_channels"] = channels

    st.subheader("Signal Preview")
    primary = sig_data if sig_data.ndim == 1 else sig_data[0, :]
    val = validate_signal_array(primary, sampling_rate=sr)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", len(primary))
    c2.metric("Duration", f"{len(primary) / sr:.2f} s")
    c3.metric("Channels", len(channels))
    c4.metric("Validation", "Valid" if val.is_valid else "Invalid")

    fig = plot_time_series(t, sig_data, channel_names=channels, title="Raw Ingested Time Series")
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------
# Section 3: Signal Analysis
# -------------------------------------------------------------
elif section == "3. Signal Analysis":
    st.title("Spectral & Time-Frequency Analysis")

    if "current_signal" not in st.session_state:
        st.warning("No signal loaded. Please navigate to 'Signal Explorer' first.")
        syn = generate_single_test_signal("forced_oscillation")
        st.session_state["current_signal"] = syn.data
        st.session_state["current_time"] = syn.time
        st.session_state["current_sr"] = 30.0
        st.session_state["current_channels"] = ["channel_1"]

    sig_data = st.session_state["current_signal"]
    t = st.session_state["current_time"]
    sr = st.session_state["current_sr"]
    channels = st.session_state["current_channels"]
    primary = sig_data if sig_data.ndim == 1 else sig_data[0, :]

    clean_sig, _rec = preprocessor.process(primary, sampling_rate=sr)
    env = extract_hilbert_envelope(clean_sig)

    st.subheader("Waveform & Envelope")
    fig_env = plot_time_series(
        t,
        clean_sig,
        channel_names=[channels[0]],
        title="Preprocessed Signal with Analytic Hilbert Envelope",
        envelope=env,
    )
    st.plotly_chart(fig_env, use_container_width=True)

    spec = analyze_spectrum(clean_sig, sampling_rate=sr)
    peak_fs = [p.frequency for p in spec.peaks]

    st.subheader("Power Spectral Density")
    fig_psd = plot_spectrum_and_psd(spec.frequencies, spec.psd, peak_frequencies=peak_fs)
    st.plotly_chart(fig_psd, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dominant Frequency", f"{spec.dominant_frequency:.2f} Hz")
    c2.metric("Spectral Concentration", f"{spec.spectral_concentration:.2f}")
    c3.metric("Spectral Entropy", f"{spec.spectral_entropy:.2f}")
    c4.metric("Spectral Centroid", f"{spec.spectral_centroid:.2f} Hz")

    st.subheader("Short-Time Fourier Transform (STFT) Spectrogram")
    stft_res = compute_stft(clean_sig, sampling_rate=sr)
    fig_spec = plot_spectrogram(stft_res.times, stft_res.frequencies, stft_res.spectrogram)
    st.plotly_chart(fig_spec, use_container_width=True)


# -------------------------------------------------------------
# Section 4: Disturbance Diagnosis
# -------------------------------------------------------------
elif section == "4. Disturbance Diagnosis":
    st.title("Disturbance Diagnosis & Source Assessment")

    if "current_signal" not in st.session_state:
        syn = generate_single_test_signal("forced_oscillation")
        st.session_state["current_signal"] = syn.data
        st.session_state["current_time"] = syn.time
        st.session_state["current_sr"] = 30.0
        st.session_state["current_channels"] = ["channel_1"]

    sig_data = st.session_state["current_signal"]
    sr = st.session_state["current_sr"]
    channels = st.session_state["current_channels"]

    diag = diagnostic_engine.diagnose_signal(sig_data, sampling_rate=sr, channel_names=channels)

    st.success(
        f"### Classification Verdict: **{diag.disturbance_type.upper()}** (Confidence: {diag.confidence * 100:.1f}%)"
    )
    st.info(f"**Diagnostic Summary:** {diag.summary}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Forced vs Natural", "Frequency Stability", "Overlapping Modes", "Source Assessment"]
    )

    with tab1:
        discrim = diag.discrimination
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Forced Likelihood", f"{discrim.forced_likelihood * 100:.0f}%")
            st.write("**Supporting Forced Evidence:**")
            for pt in discrim.forced_evidence.supporting_points:
                st.write(f"• {pt}")
        with col2:
            st.metric("Natural Ring-down Likelihood", f"{discrim.natural_likelihood * 100:.0f}%")
            st.write(
                f"**Damping Classification:** {discrim.ringdown_evidence.damping_classification} (ζ = {discrim.ringdown_evidence.damping_ratio * 100:.1f}%)"
            )
            st.write("**Supporting Natural Evidence:**")
            for pt in discrim.ringdown_evidence.supporting_points:
                st.write(f"• {pt}")

    with tab2:
        freq = diag.frequency_stability
        st.metric("Severity", freq.severity.upper())
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Deviation", f"{freq.max_frequency_deviation_hz:.3f} Hz")
        c2.metric("Peak RoCoF", f"{freq.max_rocof_hz_per_sec:.3f} Hz/s")
        c3.metric("Drift Rate", f"{freq.drift_rate_hz_per_sec:.4f} Hz/s")
        for s in freq.summary_points:
            st.write(f"• {s}")

    with tab3:
        ov = diag.overlapping_events
        st.write(f"**Multiple Modes Detected:** {'Yes' if ov.has_multiple_events else 'No'}")
        st.write(ov.summary)
        if ov.components:
            for c in ov.components:
                st.write(
                    f"• **Component {c.component_id}:** Frequency = {c.center_frequency:.2f} Hz, Energy = {c.energy_percentage:.1f}%, Sustained = {c.is_sustained}"
                )

    with tab4:
        src = diag.source_assessment
        if src.is_supported:
            st.write(f"**Candidate Source Channel:** `{src.candidate_source_channel}`")
            st.write(f"**Candidate Region:** {src.candidate_source_region}")
            st.write(f"**Assessment Confidence:** {src.confidence * 100:.0f}%")
            st.write("**Supporting Cross-Spectral Evidence:**")
            for e in src.supporting_evidence:
                st.write(f"• {e}")
        else:
            st.warning(
                "Source localization is unsupported: Requires minimum 2 synchronized measurement channels."
            )
            for lim in src.limitations:
                st.write(f"• {lim}")

    if st.button("Save Diagnostic Report to Database"):
        report = create_structured_report(diag)
        row_id = repo.save_analysis(report)
        st.success(f"Report saved to SQLite database with ID: {row_id}")


# -------------------------------------------------------------
# Section 5: Event Comparison
# -------------------------------------------------------------
elif section == "5. Event Comparison":
    st.title("Event Comparison")
    st.markdown(
        "Compare signatures, dominant frequencies, and damping between two distinct events."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Event A")
        type_a = st.selectbox(
            "Select Type A",
            ["forced_oscillation", "damped_ringdown", "poorly_damped", "overlapping_events"],
            key="type_a",
        )
        sig_a = generate_single_test_signal(type_a, random_seed=101)
        spec_a = analyze_spectrum(sig_a.data, sampling_rate=30.0)
        st.metric("Dominant Frequency A", f"{spec_a.dominant_frequency:.2f} Hz")
        fig_a = plot_time_series(sig_a.time, sig_a.data, title=f"Event A: {type_a}")
        st.plotly_chart(fig_a, use_container_width=True)

    with col2:
        st.subheader("Event B")
        type_b = st.selectbox(
            "Select Type B",
            ["damped_ringdown", "forced_oscillation", "poorly_damped", "overlapping_events"],
            key="type_b",
        )
        sig_b = generate_single_test_signal(type_b, random_seed=202)
        spec_b = analyze_spectrum(sig_b.data, sampling_rate=30.0)
        st.metric("Dominant Frequency B", f"{spec_b.dominant_frequency:.2f} Hz")
        fig_b = plot_time_series(sig_b.time, sig_b.data, title=f"Event B: {type_b}")
        st.plotly_chart(fig_b, use_container_width=True)


# -------------------------------------------------------------
# Section 6: Event Clustering
# -------------------------------------------------------------
elif section == "6. Event Clustering":
    st.title("Unsupervised Disturbance Clustering & Anomaly Discovery")
    st.markdown(
        "Cluster extracted feature vectors using PCA and KMeans/DBSCAN to discover latent event families."
    )

    num_samples = st.slider("Samples to Cluster", min_value=30, max_value=120, value=60, step=15)
    n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4)

    if st.button("Run Unsupervised Clustering"):
        with st.spinner("Extracting features and clustering..."):
            signals, df = generate_benchmark_dataset(samples_per_class=num_samples // 6)
            vec = SignalFeatureVectorizer(sampling_rate=30.0)
            x_mat = vec.fit_transform([s.data for s in signals])
            res = cluster_disturbance_features(x_mat, algorithm="kmeans", n_clusters=n_clusters)

            st.metric("Silhouette Score", f"{res.silhouette:.3f}")
            st.metric("Davies-Bouldin Index", f"{res.davies_bouldin:.3f}")
            st.metric("Anomalies Detected (Isolation Forest)", len(res.anomaly_indices))

            fig_cluster = plot_clustering_scatter(
                res.pca_2d_coordinates, res.cluster_labels, anomaly_indices=res.anomaly_indices
            )
            st.plotly_chart(fig_cluster, use_container_width=True)


# -------------------------------------------------------------
# Section 7: Model Laboratory
# -------------------------------------------------------------
elif section == "7. Model Laboratory":
    st.title("Model Laboratory & Benchmark Training")
    st.markdown(
        "Train and evaluate classical ML baselines and deep neural networks on reproducible disturbance benchmarks."
    )

    model_sel = st.selectbox(
        "Select Architecture",
        [
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "logistic_regression",
            "temporal_cnn",
            "spectrogram_cnn",
        ],
    )
    samples_per_class = st.slider(
        "Samples Per Disturbance Class", min_value=15, max_value=60, value=25, step=5
    )

    if st.button("Train and Benchmark Model"):
        with st.spinner(f"Training {model_sel}..."):
            from grid_disturbance.data.splitting import split_dataset
            from grid_disturbance.ml.training import (
                train_classical_model,
                train_deep_learning_model,
            )

            signals, df = generate_benchmark_dataset(samples_per_class=samples_per_class)
            train_df, val_df, test_df, _strat = split_dataset(df, target_column="disturbance_type")

            classes = sorted(df["disturbance_type"].unique().tolist())
            lbl_map = {c: i for i, c in enumerate(classes)}

            t_sigs = [signals[i].data for i in train_df.index]
            t_lbls = [lbl_map[train_df.loc[i, "disturbance_type"]] for i in train_df.index]
            v_sigs = [signals[i].data for i in val_df.index]
            v_lbls = [lbl_map[val_df.loc[i, "disturbance_type"]] for i in val_df.index]
            te_sigs = [signals[i].data for i in test_df.index]
            te_lbls = [lbl_map[test_df.loc[i, "disturbance_type"]] for i in test_df.index]

            if model_sel in ["temporal_cnn", "spectrogram_cnn"]:
                _trained_model, eval_report, _hist = train_deep_learning_model(
                    t_sigs,
                    t_lbls,
                    v_sigs,
                    v_lbls,
                    te_sigs,
                    te_lbls,
                    model_type=model_sel,
                    num_classes=len(classes),
                    epochs=8,
                    class_names=classes,
                )
            else:
                _trained_clf, _vec, eval_report = train_classical_model(
                    t_sigs,
                    t_lbls,
                    te_sigs,
                    te_lbls,
                    model_type=model_sel,
                    class_names=classes,
                )

            st.success(f"Benchmarked {model_sel} successfully!")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test Accuracy", f"{eval_report.accuracy:.3f}")
            c2.metric("Macro Precision", f"{eval_report.macro_precision:.3f}")
            c3.metric("Macro Recall", f"{eval_report.macro_recall:.3f}")
            c4.metric("Macro F1", f"{eval_report.macro_f1:.3f}")

            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(
                eval_report.confusion_matrix,
                index=eval_report.class_names,
                columns=eval_report.class_names,
            )
            st.dataframe(cm_df)


# -------------------------------------------------------------
# Section 8: Analysis History
# -------------------------------------------------------------
elif section == "8. Analysis History":
    st.title("Analysis History & Persistence")
    st.markdown("Reopen previous analyses, inspect past diagnostic records, and export reports.")

    analyses = repo.list_analyses(limit=25)
    if not analyses:
        st.info(
            "No saved analyses found in local database (data/storage.db). Run an analysis in 'Disturbance Diagnosis' and click 'Save Diagnostic Report'."
        )
    else:
        history_df = pd.DataFrame(analyses)
        st.dataframe(history_df)

        selected_id = st.selectbox("Select Analysis Record to View", [a["id"] for a in analyses])
        if selected_id:
            rec = repo.get_analysis_by_id(selected_id)
            if rec:
                st.subheader(f"Report Details: Event {rec.event_id}")
                st.json(rec.model_dump())
