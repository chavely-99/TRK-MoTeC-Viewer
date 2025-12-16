import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.ndimage import uniform_filter1d
import tempfile
import os

from ldparser import ldData

st.set_page_config(
    page_title="TRK MoTeC Viewer",
    page_icon="TH_FullLogo_White.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Compact styling
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0;}
    footer {visibility: hidden;}
    h1 {font-size: 1.8rem !important; margin-bottom: 0.5rem !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {padding: 6px 12px; font-size: 0.9rem;}
    div[data-testid="stVerticalBlock"] > div {gap: 0.3rem !important;}
    .stSelectbox, .stMultiSelect, .stTextInput {margin-bottom: 0.3rem !important;}
    .stSlider {padding-top: 0.5rem !important; margin-bottom: 0.3rem !important;}
    .stCheckbox {margin-bottom: 0 !important;}
    .stExpander {margin-bottom: 0.5rem !important;}
    hr {margin: 0.5rem 0 !important;}

    /* File card styling */
    .file-card {
        border-left: 3px solid;
        padding: 8px 12px;
        margin: 6px 0;
        background: rgba(255,255,255,0.02);
        border-radius: 0 4px 4px 0;
    }
    .file-card .filename {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }

    /* Lap nav button styling */
    .lap-nav-btn button {
        padding: 0 8px !important;
        min-width: 36px !important;
    }

    /* Stats bar styling */
    .stats-bar {
        font-size: 0.8rem;
        color: #888;
        padding: 4px 0;
        border-top: 1px solid #333;
        margin-top: 8px;
    }
    .stats-bar span {
        margin-right: 16px;
    }

    /* Empty state styling */
    .empty-state {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 350px;
        border: 1px dashed #444;
        border-radius: 8px;
        color: #666;
        text-align: center;
    }
    .empty-state p {
        margin: 4px 0;
    }

    /* Control panel styling */
    .ctrl-section {
        background: rgba(255,255,255,0.02);
        border-radius: 6px;
        padding: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============ UTILITY FUNCTIONS ============

def apply_filter(data, filter_type, param, sample_freq=None):
    if filter_type == "None":
        return data
    elif filter_type == "Moving Avg":
        return uniform_filter1d(data, size=max(1, int(param)), mode='nearest')
    elif filter_type == "Butterworth":
        if sample_freq and sample_freq > 0 and param > 0 and len(data) > 15:
            nyquist = sample_freq / 2
            if param >= nyquist:
                param = nyquist * 0.99
            b, a = signal.butter(4, param / nyquist, btype='low')
            return signal.filtfilt(b, a, data)
        return data
    elif filter_type == "Median":
        window = max(3, int(param))
        if window % 2 == 0: window += 1
        return signal.medfilt(data, kernel_size=window)
    elif filter_type == "Savgol":
        window = max(5, int(param))
        if window % 2 == 0: window += 1
        return signal.savgol_filter(data, window, 3)
    return data


def downsample_for_plot(x_data, y_data, max_points=50000):
    if len(x_data) <= max_points:
        return x_data, y_data
    n_bins = max_points // 2
    bin_size = len(x_data) // n_bins
    indices = []
    for i in range(n_bins):
        start = i * bin_size
        end = min(start + bin_size, len(x_data))
        if start >= len(x_data): break
        chunk = y_data[start:end]
        if len(chunk) > 0:
            min_idx = start + np.argmin(chunk)
            max_idx = start + np.argmax(chunk)
            if min_idx < max_idx:
                indices.extend([min_idx, max_idx])
            else:
                indices.extend([max_idx, min_idx])
    return x_data[indices], y_data[indices]


def find_laps(beacon_data, freq):
    """Find lap boundaries from Beacon pulse channel, including in-lap and out-lap"""
    if beacon_data is None or len(beacon_data) == 0:
        return []
    threshold = np.max(beacon_data) * 0.5
    if threshold == 0:
        return []
    pulses = np.where(np.diff((beacon_data > threshold).astype(int)) == 1)[0]

    laps = []
    total_samples = len(beacon_data)

    if len(pulses) == 0:
        return []

    # In-lap (from start to first beacon)
    if pulses[0] > 0:
        in_lap_time = pulses[0] / freq if freq > 0 else 0
        laps.append({'num': 'In', 'start': 0, 'end': pulses[0], 'time': in_lap_time, 'type': 'in'})

    # Complete laps (between beacons)
    for i, start in enumerate(pulses[:-1]):
        end = pulses[i + 1]
        lap_time = (end - start) / freq if freq > 0 else 0
        laps.append({'num': i + 1, 'start': start, 'end': end, 'time': lap_time, 'type': 'complete'})

    # Out-lap (from last beacon to end)
    if pulses[-1] < total_samples - 1:
        out_lap_time = (total_samples - pulses[-1]) / freq if freq > 0 else 0
        laps.append({'num': 'Out', 'start': pulses[-1], 'end': total_samples, 'time': out_lap_time, 'type': 'out'})

    return laps


def format_lap_time(seconds):
    """Format seconds as m:ss.xxx"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:06.3f}"


# ============ CONSTANTS ============

LINE_STYLES = {"───": "solid", "- - -": "dash", "•••": "dot", "-•-": "dashdot"}
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
               '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
FILE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# ============ SESSION STATE ============

if 'files' not in st.session_state:
    st.session_state['files'] = {}
if 'markers' not in st.session_state:
    st.session_state['markers'] = []
if 'plot_zoom' not in st.session_state:
    st.session_state['plot_zoom'] = {}
if 'time_offsets' not in st.session_state:
    st.session_state['time_offsets'] = {}
if 'lap_selections' not in st.session_state:
    st.session_state['lap_selections'] = {}


# ============ HEADER & FILE UPLOAD ============

st.markdown("# TRK MoTeC Viewer")

# Auto-collapse uploader if files already loaded
has_files = len(st.session_state.get('files', {})) > 0
with st.expander("📂 Add Data", expanded=not has_files):
    uploaded_files = st.file_uploader(
        "Drop MoTeC files",
        type=['ld', 'csv'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

# Process uploaded files
if uploaded_files:
    for uf in uploaded_files:
        if uf.name not in st.session_state['files']:
            file_ext = uf.name.split('.')[-1].lower()
            if file_ext == 'ld':
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ld') as tmp:
                    tmp.write(uf.getvalue())
                    tmp_path = tmp.name
                ld = ldData.fromfile(tmp_path)
                st.session_state['files'][uf.name] = {
                    'type': 'ld',
                    'data': {chan.name: np.array(chan.data) for chan in ld.channs},
                    'channels': [chan.name for chan in ld.channs],
                    'info': {chan.name: {'freq': chan.freq, 'unit': chan.unit} for chan in ld.channs},
                    'head': {'driver': ld.head.driver, 'vehicleid': ld.head.vehicleid,
                             'venue': ld.head.venue, 'datetime': ld.head.datetime}
                }
                os.unlink(tmp_path)
            else:
                df = pd.read_csv(uf)
                st.session_state['files'][uf.name] = {
                    'type': 'csv',
                    'data': {col: df[col].values for col in df.columns},
                    'channels': df.columns.tolist(),
                    'info': {col: {'freq': 0, 'unit': ''} for col in df.columns},
                    'head': {}
                }

# Show loaded files
files = st.session_state['files']
if files:
    file_names = list(files.keys())

    # Detect laps for each file
    file_laps = {}
    for fn in file_names:
        f = files[fn]
        if 'Beacon' in f['channels']:
            beacon_data = f['data'].get('Beacon')
            beacon_freq = f['info'].get('Beacon', {}).get('freq', 100)

            # Use highest frequency channel that has same sample count as beacon
            # to get accurate timing (beacon freq in file may be rounded/wrong)
            beacon_len = len(beacon_data)
            best_freq = beacon_freq
            for ch_name, ch_info in f['info'].items():
                ch_len = len(f['data'].get(ch_name, []))
                if ch_len == beacon_len and ch_info.get('freq', 0) > best_freq:
                    best_freq = ch_info['freq']

            file_laps[fn] = {'laps': find_laps(beacon_data, best_freq), 'beacon_freq': best_freq}
        else:
            file_laps[fn] = {'laps': [], 'beacon_freq': 100}

    # File list with lap selectors and time offsets
    for i, fn in enumerate(file_names):
        color = FILE_COLORS[i % len(FILE_COLORS)]
        laps_info = file_laps[fn]
        laps = laps_info['laps']

        # File card with colored left border
        st.markdown(f'<div class="file-card" style="border-left-color:{color};"><span class="filename">{fn}</span></div>', unsafe_allow_html=True)

        # Lap selector row for this file
        if laps:
            # Best lap only from complete laps
            complete_laps = [l for l in laps if l.get('type') == 'complete']
            best_lap = min(complete_laps, key=lambda l: l['time']) if complete_laps else None
            best_lap_idx = None

            lap_options = ["All Data"]
            for idx, l in enumerate(laps):
                lap_label = f"In ({format_lap_time(l['time'])})" if l['type'] == 'in' else \
                            f"Out ({format_lap_time(l['time'])})" if l['type'] == 'out' else \
                            f"Lap {l['num']} ({format_lap_time(l['time'])})"
                if best_lap and l.get('type') == 'complete' and l['num'] == best_lap['num']:
                    lap_label += " *"
                    best_lap_idx = idx + 1  # +1 because "All Data" is index 0
                lap_options.append(lap_label)

            # Initialize lap selection for this file
            if fn not in st.session_state['lap_selections']:
                st.session_state['lap_selections'][fn] = 0

            current_lap_idx = st.session_state['lap_selections'].get(fn, 0)

            # Layout: [Prev] [Dropdown] [Next] [Best] | [Offset]
            btn_cols = st.columns([0.6, 3, 0.6, 0.8, 0.1, 2])

            with btn_cols[0]:
                if st.button("◀", key=f"prev_{fn}", disabled=(current_lap_idx <= 0), help="Previous lap"):
                    st.session_state['lap_selections'][fn] = current_lap_idx - 1
                    st.rerun()

            with btn_cols[1]:
                lap_idx = st.selectbox(
                    f"Lap {fn}", lap_options,
                    index=current_lap_idx,
                    label_visibility="collapsed", key=f"lap_{fn}"
                )
                st.session_state['lap_selections'][fn] = lap_options.index(lap_idx)

            with btn_cols[2]:
                if st.button("▶", key=f"next_{fn}", disabled=(current_lap_idx >= len(lap_options) - 1), help="Next lap"):
                    st.session_state['lap_selections'][fn] = current_lap_idx + 1
                    st.rerun()

            with btn_cols[3]:
                if st.button("Best", key=f"best_{fn}", disabled=(best_lap_idx is None), help="Jump to fastest lap"):
                    if best_lap_idx is not None:
                        st.session_state['lap_selections'][fn] = best_lap_idx
                        st.rerun()

            # Time offset for non-primary files
            with btn_cols[5]:
                if i > 0:
                    offset = st.number_input(
                        f"Offset {fn}", value=st.session_state['time_offsets'].get(fn, 0.0),
                        step=0.1, format="%.2f", label_visibility="collapsed", key=f"offset_{fn}",
                        help="Time offset (seconds)"
                    )
                    st.session_state['time_offsets'][fn] = offset
                else:
                    st.markdown('<span style="color:#666;font-size:0.8rem;">Ref</span>', unsafe_allow_html=True)
        else:
            st.caption("No laps detected (missing Beacon channel)")

    # Clear button
    if st.button("Clear All Files", key="clear_files", type="secondary"):
        st.session_state['files'] = {}
        st.session_state['time_offsets'] = {}
        st.session_state['lap_selections'] = {}
        st.session_state['ts_channels'] = []
        st.session_state['markers'] = []
        st.rerun()

    # Use first file as primary for channel list
    primary_file = file_names[0]
    pf = files[primary_file]

    # Stats bar (cleaner format)
    head = pf.get('head', {})
    stats_parts = [f"{len(pf['channels'])} channels"]
    if head.get('vehicleid'):
        stats_parts.append(head.get('vehicleid'))
    if head.get('venue'):
        stats_parts.append(head.get('venue'))
    if head.get('driver'):
        stats_parts.append(head.get('driver'))
    st.markdown(f'<div class="stats-bar">{" · ".join(stats_parts)}</div>', unsafe_allow_html=True)

    # ============ MAIN TABS ============
    tabs = st.tabs(["Time Series", "XY Scatter"])

    # ============ TAB 1: TIME SERIES ============
    with tabs[0]:
        ctrl_col, plot_col = st.columns([1, 3])

        with ctrl_col:
            # Channel selection - keep already selected channels visible even when filtering
            search = st.text_input("Search", placeholder="Filter channels...", label_visibility="collapsed", key="ts_search")

            # Get current selections (if any) - must always be in options list
            current_selections = st.session_state.get('ts_channels', [])

            # Filter channels based on search
            if search:
                matching = [c for c in pf['channels'] if search.lower() in c.lower()]
            else:
                matching = pf['channels']

            # CRITICAL: Always include current selections in options list (at the front)
            # This prevents Streamlit from clearing selections when options change
            filtered_ch = list(current_selections) + [c for c in matching if c not in current_selections]

            selected = st.multiselect("Channels", filtered_ch, label_visibility="collapsed", key="ts_channels", placeholder="Select channels...")

            # Plot settings in expander
            with st.expander("Plot Settings", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    mode = st.radio("Mode", ["Stacked", "Overlay"], horizontal=True, label_visibility="collapsed")
                with c2:
                    x_mode = st.radio("X-Axis", ["Time", "Samples"], horizontal=True, label_visibility="collapsed")

                show_stats = st.checkbox("Show Min/Max/Avg", value=False, key="show_stats")

                ftype = st.selectbox("Filter", ["None", "Moving Avg", "Butterworth", "Median", "Savgol"], key="filter_type")
                if ftype != "None":
                    if ftype == "Moving Avg":
                        fparam = st.slider("Window", 3, 101, 11, 2, label_visibility="collapsed")
                    elif ftype == "Butterworth":
                        fparam = st.slider("Cutoff Hz", 0.5, 50.0, 5.0, 0.5, label_visibility="collapsed")
                    else:
                        fparam = st.slider("Window", 3, 51, 5, 2, label_visibility="collapsed")
                    show_raw = st.checkbox("Show raw", True)
                else:
                    fparam, show_raw = 0, False

            # Markers (horizontal only)
            with st.expander("Markers", expanded=False):
                m_name = st.text_input("Label", placeholder="e.g., Max Throttle", label_visibility="collapsed", key="m_name")

                mc1, mc2, mc3 = st.columns([2, 1.5, 0.5])
                with mc1:
                    m_val = st.number_input("Value", value=0.0, format="%.2f", label_visibility="collapsed", key="m_val")
                with mc2:
                    m_style = st.selectbox("Style", list(LINE_STYLES.keys()), label_visibility="collapsed", key="m_style")
                with mc3:
                    m_color = st.color_picker("Color", "#FF0000", label_visibility="collapsed", key="m_color")

                # Target channel selector
                marker_targets = ["All Charts"] + (selected if selected else [])
                m_target = st.selectbox("Apply to", marker_targets, label_visibility="collapsed", key="m_target")

                if st.button("➕ Add Marker", key="add_marker"):
                    if m_name:
                        st.session_state['markers'].append({
                            'name': m_name, 'type': 'Horizontal',
                            'value': m_val, 'color': m_color, 'style': m_style,
                            'target': m_target
                        })

                if st.session_state['markers']:
                    st.markdown("---")
                    for i, m in enumerate(st.session_state['markers']):
                        mc1, mc2 = st.columns([5, 1])
                        with mc1:
                            target_str = f" → {m.get('target', 'All')}" if m.get('target') and m.get('target') != "All Charts" else ""
                            st.caption(f"─ **{m['name']}**: {m['value']:.2f}{target_str}")
                        with mc2:
                            if st.button("✕", key=f"dm_{i}"):
                                st.session_state['markers'].pop(i)
                                st.rerun()

        # Plot area
        with plot_col:
            if selected:
                # Prepare data from all files
                all_plot_data = {}
                for fi, fn in enumerate(file_names):
                    f = files[fn]
                    file_data = {}

                    # Get lap range for THIS file
                    lap_info = file_laps[fn]
                    laps = lap_info['laps']
                    beacon_freq = lap_info['beacon_freq']
                    lap_idx = st.session_state['lap_selections'].get(fn, 0)
                    lap_range = None
                    if lap_idx > 0 and lap_idx <= len(laps):
                        lap = laps[lap_idx - 1]
                        lap_range = (lap['start'], lap['end'])

                    for ch in selected:
                        if ch in f['data']:
                            raw = f['data'][ch]
                            freq = f['info'][ch]['freq']

                            # Apply lap range filter for THIS file
                            if lap_range:
                                start_idx, end_idx = lap_range
                                # Scale indices if channel has different frequency
                                if freq > 0 and beacon_freq > 0:
                                    scale = freq / beacon_freq
                                    ch_start = int(start_idx * scale)
                                    ch_end = int(end_idx * scale)
                                else:
                                    ch_start, ch_end = start_idx, end_idx
                                raw = raw[ch_start:ch_end]

                            filt = apply_filter(raw, ftype, fparam, freq)
                            # X-axis starts at 0 for lap view, apply time offset for non-primary files
                            x = np.arange(len(raw)) / freq if x_mode == "Time" and freq > 0 else np.arange(len(raw))
                            # Apply time offset (only in Time mode)
                            if x_mode == "Time":
                                time_offset = st.session_state['time_offsets'].get(fn, 0.0)
                                x = x + time_offset
                            file_data[ch] = {'x': x, 'raw': raw, 'filt': filt,
                                             'unit': f['info'][ch]['unit'], 'freq': freq}
                    all_plot_data[fn] = file_data

                multi_file = len(file_names) > 1

                if mode == "Overlay":
                    fig = go.Figure()
                    y_min_all, y_max_all = float('inf'), float('-inf')
                    x_min_all, x_max_all = float('inf'), float('-inf')

                    for fi, fn in enumerate(file_names):
                        file_color = FILE_COLORS[fi % len(FILE_COLORS)] if multi_file else None
                        for ci, ch in enumerate(selected):
                            if ch not in all_plot_data[fn]:
                                continue
                            d = all_plot_data[fn][ch]
                            color = file_color or PLOT_COLORS[ci % len(PLOT_COLORS)]

                            if ftype != "None" and show_raw:
                                xd, yd = downsample_for_plot(d['x'], d['raw'])
                                fig.add_trace(go.Scattergl(x=xd, y=yd,
                                    name=f"{fn}: {ch} (raw)" if multi_file else f"{ch} (raw)",
                                    line=dict(color=color, width=1), opacity=0.3))

                            y = d['filt'] if ftype != "None" else d['raw']
                            xd, yd = downsample_for_plot(d['x'], y)
                            y_min_all, y_max_all = min(y_min_all, np.nanmin(yd)), max(y_max_all, np.nanmax(yd))
                            x_min_all, x_max_all = min(x_min_all, xd.min()), max(x_max_all, xd.max())

                            fig.add_trace(go.Scattergl(x=xd, y=yd,
                                name=f"{fn}: {ch}" if multi_file else f"{ch} [{d['unit']}]",
                                line=dict(color=color, width=2)))

                    # Add horizontal markers (filter by target)
                    for m in st.session_state['markers']:
                        target = m.get('target', 'All Charts')
                        # Show marker if target is All Charts or if target channel is selected
                        if target == "All Charts" or target in selected:
                            fig.add_trace(go.Scatter(x=[x_min_all, x_max_all], y=[m['value']]*2, mode='lines',
                                name=m['name'], line=dict(color=m['color'], width=2, dash=LINE_STYLES[m['style']])))

                    # Create stable uirevision based on channels and lap (not markers)
                    ui_rev = f"{'-'.join(selected)}_{st.session_state.get('current_lap', 0)}"
                    fig.update_layout(height=max(500, 100*len(selected)),
                        xaxis_title="Time (s)" if x_mode == "Time" else "Samples",
                        legend=dict(orientation="h", y=1.02), hovermode='x unified',
                        uirevision=ui_rev, margin=dict(l=50, r=10, t=30, b=40))
                    # Set uirevision on axes to preserve zoom
                    fig.update_xaxes(uirevision=ui_rev)
                    fig.update_yaxes(uirevision=ui_rev)
                    st.plotly_chart(fig, use_container_width=True, key="ts_overlay")

                else:  # Stacked
                    fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True, vertical_spacing=0.02)

                    for ci, ch in enumerate(selected):
                        for fi, fn in enumerate(file_names):
                            if ch not in all_plot_data[fn]:
                                continue
                            d = all_plot_data[fn][ch]
                            color = FILE_COLORS[fi % len(FILE_COLORS)] if multi_file else PLOT_COLORS[ci % len(PLOT_COLORS)]

                            if ftype != "None" and show_raw:
                                xd, yd = downsample_for_plot(d['x'], d['raw'])
                                fig.add_trace(go.Scattergl(x=xd, y=yd,
                                    name=f"{fn}: {ch} (raw)" if multi_file else f"{ch} (raw)",
                                    line=dict(color=color, width=1), opacity=0.3,
                                    showlegend=(ci == 0)), row=ci+1, col=1)

                            y = d['filt'] if ftype != "None" else d['raw']
                            xd, yd = downsample_for_plot(d['x'], y)
                            fig.add_trace(go.Scattergl(x=xd, y=yd,
                                name=f"{fn}: {ch}" if multi_file else ch,
                                line=dict(color=color, width=2),
                                showlegend=multi_file), row=ci+1, col=1)

                            # Add min/max/avg stats as horizontal lines
                            if show_stats and fi == 0:  # Only show stats for primary file
                                y_min = np.nanmin(y)
                                y_max = np.nanmax(y)
                                y_avg = np.nanmean(y)
                                x_start, x_end = d['x'].min(), d['x'].max()

                                # Min line (green, dashed)
                                fig.add_trace(go.Scatter(
                                    x=[x_start, x_end], y=[y_min, y_min], mode='lines',
                                    line=dict(color='#2ca02c', width=1, dash='dot'),
                                    name=f"Min: {y_min:.2f}", showlegend=False,
                                    hoverinfo='skip'
                                ), row=ci+1, col=1)

                                # Max line (red, dashed)
                                fig.add_trace(go.Scatter(
                                    x=[x_start, x_end], y=[y_max, y_max], mode='lines',
                                    line=dict(color='#d62728', width=1, dash='dot'),
                                    name=f"Max: {y_max:.2f}", showlegend=False,
                                    hoverinfo='skip'
                                ), row=ci+1, col=1)

                                # Avg line (orange, dashed)
                                fig.add_trace(go.Scatter(
                                    x=[x_start, x_end], y=[y_avg, y_avg], mode='lines',
                                    line=dict(color='#ff7f0e', width=1, dash='dash'),
                                    name=f"Avg: {y_avg:.2f}", showlegend=False,
                                    hoverinfo='skip'
                                ), row=ci+1, col=1)

                                # Add annotations on right side
                                fig.add_annotation(x=x_end, y=y_min, text=f"min:{y_min:.1f}",
                                    xanchor='left', showarrow=False, font=dict(size=9, color='#2ca02c'),
                                    row=ci+1, col=1)
                                fig.add_annotation(x=x_end, y=y_max, text=f"max:{y_max:.1f}",
                                    xanchor='left', showarrow=False, font=dict(size=9, color='#d62728'),
                                    row=ci+1, col=1)
                                fig.add_annotation(x=x_end, y=y_avg, text=f"avg:{y_avg:.1f}",
                                    xanchor='left', showarrow=False, font=dict(size=9, color='#ff7f0e'),
                                    row=ci+1, col=1)

                        # Y-axis label
                        unit = all_plot_data[file_names[0]].get(ch, {}).get('unit', '')
                        fig.update_yaxes(title_text=f"{ch} [{unit}]", row=ci+1, col=1)

                    # Add horizontal markers to targeted subplots
                    added = set()
                    for m in st.session_state['markers']:
                        target = m.get('target', 'All Charts')
                        for ci, ch in enumerate(selected):
                            # Skip if marker targets a specific channel and this isn't it
                            if target != "All Charts" and target != ch:
                                continue

                            d = all_plot_data[file_names[0]].get(ch)
                            if not d: continue

                            show_leg = m['name'] not in added
                            added.add(m['name'])

                            fig.add_trace(go.Scatter(x=[d['x'].min(), d['x'].max()], y=[m['value']]*2, mode='lines',
                                name=m['name'], line=dict(color=m['color'], width=2, dash=LINE_STYLES[m['style']]),
                                showlegend=show_leg), row=ci+1, col=1)

                    # Create stable uirevision based on channels and lap (not markers)
                    ui_rev = f"{'-'.join(selected)}_{st.session_state.get('current_lap', 0)}"
                    fig.update_layout(height=max(500, 200*len(selected)), showlegend=True,
                        legend=dict(orientation="h", y=1.02), hovermode='x unified',
                        uirevision=ui_rev, margin=dict(l=60, r=70 if show_stats else 10, t=30, b=40))
                    # Set uirevision on all axes to preserve zoom
                    fig.update_xaxes(uirevision=ui_rev)
                    fig.update_yaxes(uirevision=ui_rev)
                    fig.update_xaxes(title_text="Time (s)" if x_mode == "Time" else "Samples", row=len(selected), col=1)
                    st.plotly_chart(fig, use_container_width=True, key="ts_stacked")

            else:
                st.markdown('<div class="empty-state"><div><p style="font-size:1.2rem;">Select channels to plot</p><p style="font-size:0.85rem;">Use the search box to find channels</p></div></div>', unsafe_allow_html=True)

    # ============ TAB 2: XY SCATTER ============
    with tabs[1]:
        ctrl_col, plot_col = st.columns([1, 3])

        with ctrl_col:
            # Search filter for XY channels
            xy_search = st.text_input("Search", placeholder="Filter channels...", label_visibility="collapsed", key="xy_search")
            xy_filtered = [c for c in pf['channels'] if xy_search.lower() in c.lower()] if xy_search else pf['channels']

            st.caption("X-Axis")
            # Keep current selection in list if it exists
            x_current = st.session_state.get('xy_x', None)
            x_options = xy_filtered if not x_current or x_current in xy_filtered else [x_current] + xy_filtered
            x_ch = st.selectbox("X Channel", x_options, label_visibility="collapsed", key="xy_x")

            st.caption("Y-Axis")
            y_current = st.session_state.get('xy_y', None)
            y_options = xy_filtered if not y_current or y_current in xy_filtered else [y_current] + xy_filtered
            y_ch = st.selectbox("Y Channel", y_options, label_visibility="collapsed", key="xy_y")

            st.markdown("---")

            st.caption("Color by")
            color_opt = st.selectbox("Color by", ["None", "Time", "Speed"] +
                ([c for c in pf['channels'] if c not in [x_ch, y_ch]]), key="xy_color", label_visibility="collapsed")

            st.caption("Point size")
            marker_size = st.slider("Point size", 1, 10, 3, key="xy_size", label_visibility="collapsed")

        with plot_col:
            if x_ch and y_ch:
                fig = go.Figure()

                for fi, fn in enumerate(file_names):
                    f = files[fn]
                    if x_ch not in f['data'] or y_ch not in f['data']:
                        continue

                    x_data = f['data'][x_ch]
                    y_data = f['data'][y_ch]

                    # Determine color
                    if color_opt == "None":
                        color = FILE_COLORS[fi % len(FILE_COLORS)]
                        fig.add_trace(go.Scattergl(x=x_data, y=y_data, mode='markers',
                            name=fn if len(file_names) > 1 else f"{x_ch} vs {y_ch}",
                            marker=dict(size=marker_size, color=color, opacity=0.6)))
                    elif color_opt == "Time":
                        freq = f['info'][x_ch].get('freq', 100)
                        time_data = np.arange(len(x_data)) / freq if freq > 0 else np.arange(len(x_data))
                        fig.add_trace(go.Scattergl(x=x_data, y=y_data, mode='markers',
                            name=fn if len(file_names) > 1 else f"{x_ch} vs {y_ch}",
                            marker=dict(size=marker_size, color=time_data, colorscale='Viridis',
                                       colorbar=dict(title="Time (s)"), opacity=0.6)))
                    elif color_opt in f['data']:
                        color_data = f['data'][color_opt]
                        fig.add_trace(go.Scattergl(x=x_data, y=y_data, mode='markers',
                            name=fn if len(file_names) > 1 else f"{x_ch} vs {y_ch}",
                            marker=dict(size=marker_size, color=color_data, colorscale='Viridis',
                                       colorbar=dict(title=color_opt), opacity=0.6)))

                x_unit = pf['info'][x_ch]['unit']
                y_unit = pf['info'][y_ch]['unit']
                fig.update_layout(
                    height=600,
                    xaxis_title=f"{x_ch} [{x_unit}]" if x_unit else x_ch,
                    yaxis_title=f"{y_ch} [{y_unit}]" if y_unit else y_ch,
                    legend=dict(orientation="h", y=1.02),
                    uirevision="stable",
                    margin=dict(l=60, r=10, t=30, b=40)
                )
                st.plotly_chart(fig, use_container_width=True, key="xy_plot")
            else:
                st.markdown('<div class="empty-state"><div><p style="font-size:1.2rem;">Select X and Y channels</p><p style="font-size:0.85rem;">Choose channels from the dropdowns</p></div></div>', unsafe_allow_html=True)

else:
    st.markdown('''
    <div style="text-align:center;padding:2rem;">
        <h3 style="color:#888;font-weight:400;">Drop MoTeC files to start</h3>
        <p style="color:#666;font-size:0.9rem;">.ld (native) or .csv (exported) · Multiple files supported</p>
    </div>
    ''', unsafe_allow_html=True)
