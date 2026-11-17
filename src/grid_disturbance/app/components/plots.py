"""Interactive Plotly visualization components for the Streamlit diagnostic app."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_time_series(
    time: np.ndarray,
    data: np.ndarray,
    channel_names: list[str] | None = None,
    title: str = "Synchrophasor Time Series Waveform",
    envelope: np.ndarray | None = None,
) -> go.Figure:
    """Create interactive multi-channel or single-channel time series plot with optional envelope."""
    fig = go.Figure()

    if data.ndim == 1:
        fig.add_trace(
            go.Scatter(
                x=time,
                y=data,
                mode="lines",
                name=channel_names[0] if channel_names else "Signal",
                line=dict(color="#2563eb", width=1.5),
            )
        )
        if envelope is not None:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=envelope,
                    mode="lines",
                    name="Hilbert Envelope",
                    line=dict(color="#dc2626", width=2, dash="dash"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=-envelope,
                    mode="lines",
                    name="-Envelope",
                    line=dict(color="#dc2626", width=1.5, dash="dash"),
                    showlegend=False,
                )
            )
    else:
        colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
        for i in range(data.shape[0]):
            ch = (
                channel_names[i]
                if (channel_names and i < len(channel_names))
                else f"Channel {i + 1}"
            )
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=data[i, :],
                    mode="lines",
                    name=ch,
                    line=dict(color=colors[i % len(colors)], width=1.5),
                )
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude / Frequency Deviation (Hz or p.u.)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def plot_spectrum_and_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    peak_frequencies: list[float] | None = None,
    title: str = "Power Spectral Density (Welch)",
) -> go.Figure:
    """Create PSD plot with marked dominant peaks."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=psd,
            mode="lines",
            name="PSD (V²/Hz)",
            line=dict(color="#059669", width=2),
        )
    )

    if peak_frequencies:
        peak_y = [float(psd[np.argmin(np.abs(frequencies - f))]) for f in peak_frequencies]
        fig.add_trace(
            go.Scatter(
                x=peak_frequencies,
                y=peak_y,
                mode="markers+text",
                name="Identified Modes",
                text=[f"{f:.2f} Hz" for f in peak_frequencies],
                textposition="top center",
                marker=dict(color="#dc2626", size=9, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Spectral Density",
        xaxis=dict(range=[0, 10]),  # Typical grid oscillation range
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_spectrogram(
    times: np.ndarray,
    frequencies: np.ndarray,
    spectrogram: np.ndarray,
    title: str = "Short-Time Fourier Transform (STFT) Spectrogram",
) -> go.Figure:
    """Create 2D interactive spectrogram heatmap."""
    # Convert to log scale (dB)
    spec_db = 10.0 * np.log10(np.maximum(spectrogram, 1e-12))
    # Filter to 0 - 10 Hz
    mask = frequencies <= 10.0
    f_sub = frequencies[mask]
    s_sub = spec_db[mask, :]

    fig = go.Figure(
        data=go.Heatmap(
            z=s_sub,
            x=times,
            y=f_sub,
            colorscale="Viridis",
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def plot_clustering_scatter(
    coords_2d: list[list[float]],
    labels: list[int],
    anomaly_indices: list[int] | None = None,
) -> go.Figure:
    """Plot 2D PCA cluster projection."""
    fig = go.Figure()
    coords = np.array(coords_2d)

    unique_labels = sorted(list(set(labels)))
    for lbl in unique_labels:
        mask = np.array(labels) == lbl
        name = f"Cluster {lbl}" if lbl != -1 else "Noise / Unassigned"
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0], y=coords[mask, 1], mode="markers", name=name, marker=dict(size=8)
            )
        )

    if anomaly_indices:
        anom_coords = coords[anomaly_indices]
        fig.add_trace(
            go.Scatter(
                x=anom_coords[:, 0],
                y=anom_coords[:, 1],
                mode="markers",
                name="Anomalies (Isolation Forest)",
                marker=dict(symbol="x", color="#ef4444", size=11, line=dict(width=2)),
            )
        )

    fig.update_layout(
        title="2D PCA Latent Projection of Disturbance Events",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig
