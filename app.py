"""
app.py
======
Streamlit dashboard for HH-MPNN ACOPF surrogate inference on the 118-bus
system. All 35040 yearly scenarios (one per 15 minutes) are precomputed by
precompute_year.py; this app serves the results IN ORDER, advancing to the
next scenario every 15 minutes of wall-clock time.

Run with:
    streamlit run app.py

Tabs:
  1. Grid & Dispatch — 118-bus graph, predicted Pg shown in green on each
     generator bus; branches with flow violations drawn in red.
  2. Violations     — power balance mismatches (MW / MVar) and branch flow
                      violations (MVA), styled red.
  3. Cost           — dispatch cost in dollars, styled green.
"""

import time
import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYSTEM_SIZE = 118
RESULTS_PATH = f'./results_{SYSTEM_SIZE}_year.npz'

# Where to fetch the precomputed results file if it is not present locally
# (Streamlit Community Cloud deployment: the .npz is too large for a normal
# GitHub repo file, so it is attached to a GitHub RELEASE and downloaded on
# first startup).
RESULTS_URL = ('https://github.com/<user>/<repo>/releases/download/'
               f'v1.0/results_{SYSTEM_SIZE}_year.npz')

STEP_MINUTES = 15                       # one scenario per 15 minutes
YEAR_START = datetime.datetime(2026, 1, 1, 0, 0)  # timestamp of scenario 0

# Timezone used to anchor the scenario index to the wall-clock time of year.
# The load profile is Romanian, so Romanian local time makes the daily/seasonal
# pattern line up physically (HF Spaces servers run in UTC otherwise).
from zoneinfo import ZoneInfo
TIMEZONE = ZoneInfo('Europe/Bucharest')

# TODO: replace with the actual link to the TGGNN4ACOPF paper
PAPER_URL = 'https://www.sciencedirect.com/science/article/pii/S2666546826001680'

# Base power for p.u. -> physical unit conversion (MW / MVar / MVA).
# 100 MVA is the standard base for the IEEE-118 / PGLearn datasets.
BASE_MVA = 100.0

GREEN = '#1db954'   # dispatch / cost colour
RED = '#e63946'     # violations colour
GRAY = '#8d99ae'    # neutral grid elements

# Shared legend style: slightly larger font, kept at the right-hand side of
# each plot but vertically CENTERED — pulled down from Plotly's default
# top-right corner so it never clashes with the Streamlit/Plotly toolbar
# icons that sit at the top of every chart.
LEGEND_STYLE = dict(
    font=dict(size=14),
    orientation='v',          # vertical list of entries
    yanchor='middle', y=0.5,  # centered vertically alongside the plot
    xanchor='left', x=1.02,   # just outside the right edge of the plot area
)

st.set_page_config(page_title='118-Bus GNN ACOPF Monitor', layout='wide')


# ---------------------------------------------------------------------------
# Data loading — cached so the .npz is read from disk only once per session
# ---------------------------------------------------------------------------
@st.cache_resource
def load_results(path):
    """Load the precomputed yearly results file into memory once.
    On a fresh deployment (e.g. Streamlit Community Cloud) the file is not
    in the repo; download it from the GitHub release on first startup."""
    import os
    import urllib.request

    if not os.path.exists(path):
        with st.spinner('First startup: downloading precomputed results '
                        '(one-off, a few hundred MB)...'):
            # Download to a temp name, then rename — avoids a half-written
            # file being treated as valid if the download is interrupted
            tmp = path + '.part'
            urllib.request.urlretrieve(RESULTS_URL, tmp)
            os.replace(tmp, path)

    data = np.load(path)
    # Materialise into a plain dict of arrays (npz lazy-loads otherwise)
    return {k: data[k] for k in data.files}


try:
    R = load_results(RESULTS_PATH)
except FileNotFoundError:
    st.error(f'Results file not found: {RESULTS_PATH}. '
             'Run precompute_year.py first.')
    st.stop()

N_SCENARIOS = R['cost'].shape[0]        # 35040 for a full year
N_BRANCH = R['branch_list'].shape[0]
GEN_BUSES = R['generator_indices']      # bus index of each generator


# ---------------------------------------------------------------------------
# Time-step logic: serve scenarios in order, one every 15 minutes.
# The session anchors t0 at first load; index = elapsed // 15min (mod N).
# ---------------------------------------------------------------------------
if 't0' not in st.session_state:
    st.session_state.t0 = time.time()   # wall-clock anchor of scenario 0

with st.sidebar:
    st.header('Playback')

    mode = st.radio('Mode', ['Auto (every 15 min)', 'Manual browse'],
                    help='Auto advances one scenario per 15 real minutes; '
                         'Manual lets you inspect any interval of the year.')

    # Accelerated playback is useful for demos/testing without waiting 15 min
    demo = st.checkbox('Demo speed (advance every 10 s)', value=False)
    step_seconds = 10 if demo else STEP_MINUTES * 60

    start_idx = st.number_input('Start scenario index', min_value=0,
                                max_value=N_SCENARIOS - 1, value=0, step=1,
                                help='Used by Manual browse and Demo speed; '
                                     'Auto mode follows the time of year.')

    if mode == 'Auto (every 15 min)':
        if demo:
            # Demo playback: session-anchored, advancing every 10 s from
            # start_idx — for testing without waiting real 15-min intervals
            elapsed = time.time() - st.session_state.t0
            t_idx = (int(start_idx) + int(elapsed // step_seconds)) % N_SCENARIOS
            remaining = step_seconds - (elapsed % step_seconds)
        else:
            # CHANGED: index anchored to the actual wall-clock time of year
            # (in the profile's timezone), so every visitor sees "now" in the
            # yearly profile regardless of when their session started, and a
            # page reload no longer restarts playback. start_idx is ignored
            # in this mode.
            now = datetime.datetime.now(TIMEZONE)
            minute_of_year = ((now - datetime.datetime(now.year, 1, 1,
                                                       tzinfo=TIMEZONE))
                              .total_seconds() / 60.0)
            # Modulo guards the tail of leap years (day 366 wraps to Jan 1,
            # since the profile only covers 365 days = 35040 intervals)
            t_idx = int(minute_of_year // STEP_MINUTES) % N_SCENARIOS
            # Seconds until the next quarter-hour boundary
            remaining = step_seconds - (minute_of_year * 60.0) % step_seconds

        # Schedule an automatic rerun exactly when the next interval starts
        try:
            # Preferred: non-blocking autorefresh (pip install streamlit-autorefresh)
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=int(remaining * 1000) + 500, key='ticker')
        except ImportError:
            st.caption('Install `streamlit-autorefresh` for automatic updates; '
                       'until then, press the button when the next interval is due.')
            st.button('Refresh now')
        st.metric('Next update in', f'{int(remaining // 60)} m {int(remaining % 60)} s')
    else:
        # Manual browsing over the whole year
        t_idx = st.slider('Scenario index', 0, N_SCENARIOS - 1, int(start_idx))

    st.caption('Yearly profile: 15 min resolution')

# Human-readable timestamp of the current scenario
sim_time = YEAR_START + datetime.timedelta(minutes=STEP_MINUTES * t_idx)

# Main title with embedded link to the paper, plus subtitle
st.title('HH-MPNN ACOPF Decision Support Tool')
st.markdown(f'[Link to our paper]({PAPER_URL})')
st.markdown('*Realistic yearly load profile from the Romanian grid '
            'scaled for the IEEE-118 bus grid*')
st.subheader(f'Scenario {t_idx} — {sim_time:%A %d %B, %H:%M}')


# ---------------------------------------------------------------------------
# Slice out the current scenario's results
# ---------------------------------------------------------------------------
p_now = R['p_pred'][t_idx]              # (n_gen, 2)  [Pg, Qg] p.u.
pbal_p_now = R['pbal_p'][t_idx]         # (118,) active mismatch p.u.
pbal_q_now = R['pbal_q'][t_idx]         # (118,) reactive mismatch p.u.
fwd_viol_now = R['fwd_viol'][t_idx]     # (n_branch,) p.u.
rev_viol_now = R['rev_viol'][t_idx]     # (n_branch,) p.u.
cost_now = float(R['cost'][t_idx])      # $
branch_viol_now = np.maximum(fwd_viol_now, rev_viol_now)  # worst direction per branch


# ---------------------------------------------------------------------------
# Grid figure builder
# ---------------------------------------------------------------------------
def build_grid_figure():
    """118-bus graph: gray non-generator buses, green generator buses with the
    predicted Pg annotated above them, violated branches drawn in red."""
    xy = R['node_xy']                   # (118, 2) precomputed layout
    branches = R['branch_list']         # (n_branch, 2)

    fig = go.Figure()

    # --- branches: one trace for healthy (gray), one for violated (red) ---
    for violated, colour, width, name in [(False, GRAY, 1.0, 'branch'),
                                          (True, RED, 2.5, 'violated branch')]:
        xs, ys = [], []
        for b, (i, j) in enumerate(branches):
            if (branch_viol_now[b] > 1e-6) != violated:
                continue
            xs += [xy[i, 0], xy[j, 0], None]   # None breaks the line segment
            ys += [xy[i, 1], xy[j, 1], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=name,
                                     line=dict(color=colour, width=width),
                                     hoverinfo='skip'))

    # --- non-generator buses (gray dots) ---
    gen_set = set(int(b) for b in GEN_BUSES)
    other = [i for i in range(SYSTEM_SIZE) if i not in gen_set]
    fig.add_trace(go.Scatter(
        x=xy[other, 0], y=xy[other, 1], mode='markers', name='bus',
        marker=dict(size=6, color=GRAY),
        text=[f'Bus {i}' for i in other], hoverinfo='text'))

    # --- generator buses (green, Pg annotated on top of the node) ---
    # Multiple generators can share a bus: sum their Pg for the label
    pg_by_bus = {}
    for g, bus in enumerate(GEN_BUSES):
        pg_by_bus[int(bus)] = pg_by_bus.get(int(bus), 0.0) + float(p_now[g, 0])

    gen_x = [xy[b, 0] for b in pg_by_bus]
    gen_y = [xy[b, 1] for b in pg_by_bus]
    labels = [f'{pg:.2f}' for pg in pg_by_bus.values()]
    hover = [f'Bus {b}<br>Pg = {pg:.3f} p.u.' for b, pg in pg_by_bus.items()]

    fig.add_trace(go.Scatter(
        x=gen_x, y=gen_y, mode='markers+text', name='generator bus',
        marker=dict(size=11, color=GREEN, symbol='square'),
        text=labels, textposition='top center',
        textfont=dict(color=GREEN, size=11),
        hovertext=hover, hoverinfo='text'))

    fig.update_layout(
        showlegend=True, legend=LEGEND_STYLE,
        height=650, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor='white')
    return fig


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_grid, tab_viol, tab_cost = st.tabs(
    ['🗺️ Grid & Dispatch', '🚨 Violations', '💰 Cost'])

# --- Tab 1: grid with dispatch --------------------------------------------
with tab_grid:
    st.markdown(f'Predicted generator dispatch shown in '
                f'<span style="color:{GREEN}"><b>green</b></span> (Pg, p.u.) '
                f'above each generator bus. Branches violating their rating '
                f'are drawn in <span style="color:{RED}"><b>red</b></span>.',
                unsafe_allow_html=True)
    st.plotly_chart(build_grid_figure(), use_container_width=True)

    # Sum of predicted Pg over all generators, converted from p.u. to MW
    total_pg_mw = float(p_now[:, 0].sum()) * BASE_MVA
    st.markdown(f'<h4 style="color:{GREEN}">Total predicted dispatch: '
                f'{total_pg_mw:,.1f} MW</h4>', unsafe_allow_html=True)

# --- Tab 2: violations (red) ----------------------------------------------
with tab_viol:
    st.markdown(f'<h3 style="color:{RED}">Constraint violations</h3>',
                unsafe_allow_html=True)

    # Convert p.u. results to physical units for display:
    #   active balance -> MW, reactive balance -> MVar,
    #   branch |S| violations -> MVA (apparent power)
    pbal_p_mw = pbal_p_now * BASE_MVA
    pbal_q_mvar = pbal_q_now * BASE_MVA
    fwd_viol_mva = fwd_viol_now * BASE_MVA
    rev_viol_mva = rev_viol_now * BASE_MVA
    branch_viol_mva = branch_viol_now * BASE_MVA

    # Headline metrics, all in red
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ('Max |active balance mismatch|', np.abs(pbal_p_mw).max(), 'MW'),
        ('Max |reactive balance mismatch|', np.abs(pbal_q_mvar).max(), 'MVar'),
        ('Max forward flow violation', fwd_viol_mva.max(), 'MVA'),
        ('Max reverse flow violation', rev_viol_mva.max(), 'MVA'),
    ]
    for col, (label, value, unit) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f'<div style="color:{RED}"><b>{label}</b><br>'
                     f'<span style="font-size:1.6em">{value:,.2f} {unit}</span></div>',
                     unsafe_allow_html=True)

    st.divider()

    left, right = st.columns(2)

    # Per-bus power balance mismatch bars
    with left:
        st.markdown(f'<b style="color:{RED}">Power balance mismatch per bus</b>',
                    unsafe_allow_html=True)
        fig_pb = go.Figure()
        fig_pb.add_trace(go.Bar(x=list(range(SYSTEM_SIZE)), y=pbal_p_mw,
                                name='Active (MW)', marker_color=RED))
        fig_pb.add_trace(go.Bar(x=list(range(SYSTEM_SIZE)), y=pbal_q_mvar,
                                name='Reactive (MVar)', marker_color='#f4a3ab'))
        fig_pb.update_layout(barmode='group', height=380, legend=LEGEND_STYLE,
                             xaxis_title='Bus', yaxis_title='Mismatch (MW / MVar)',
                             margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pb, use_container_width=True)

    # Per-branch flow violation bars (worst of the two directions)
    with right:
        st.markdown(f'<b style="color:{RED}">Branch flow violations '
                    f'(worst direction)</b>', unsafe_allow_html=True)
        fig_bf = go.Figure()
        fig_bf.add_trace(go.Bar(x=list(range(N_BRANCH)), y=branch_viol_mva,
                                marker_color=RED, name='|S| − rating'))
        fig_bf.update_layout(height=380, legend=LEGEND_STYLE,
                             xaxis_title='Branch index',
                             yaxis_title='Violation (MVA)',
                             margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bf, use_container_width=True)

    n_viol_branches = int((branch_viol_now > 1e-6).sum())
    st.markdown(f'<span style="color:{RED}">{n_viol_branches} of {N_BRANCH} '
                f'branches above their long-term rating at this interval.</span>',
                unsafe_allow_html=True)

# --- Tab 3: cost (green) ---------------------------------------------------
with tab_cost:
    st.markdown(f'<h3 style="color:{GREEN}">Cost of predicted dispatch</h3>',
                unsafe_allow_html=True)

    st.markdown(f'<div style="color:{GREEN}; font-size:3em; font-weight:bold">'
                f'${cost_now:,.2f}</div>', unsafe_allow_html=True)

    # Cost trajectory of the year so far, current interval highlighted
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=list(range(t_idx + 1)), y=R['cost'][:t_idx + 1],
        mode='lines', line=dict(color=GREEN, width=1.5), name='cost'))
    fig_cost.add_trace(go.Scatter(
        x=[t_idx], y=[cost_now], mode='markers',
        marker=dict(color=GREEN, size=10), name='now'))
    fig_cost.update_layout(height=420, legend=LEGEND_STYLE,
                           xaxis_title='Scenario index (15-min intervals)',
                           yaxis_title='Dispatch cost ($)',
                           margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_cost, use_container_width=True)

    # Simple running statistics over the served portion of the year
    served = R['cost'][:t_idx + 1]
    a, b, c = st.columns(3)
    a.metric('Mean cost so far', f'${served.mean():,.2f}')
    b.metric('Min cost so far', f'${served.min():,.2f}')
    c.metric('Max cost so far', f'${served.max():,.2f}')
