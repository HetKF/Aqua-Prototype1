"""
AquaShield - Steps 16-18: Streamlit Dashboard

Ties together everything built so far into one visual demo:
  - Normal operation view
  - Inject Contamination -> source localization + containment options
  - Maintenance scenario -> before/after comparison

Run (same folder as aquashield.inp):
    streamlit run app.py
"""

import itertools
import math

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import wntr

st.set_page_config(page_title="AquaShield", layout="wide")

# ---------------------------------------------------------------------------
# Shared prototype constants (all synthetic / demo assumptions, clearly labeled)
# ---------------------------------------------------------------------------
QUALITY_ALERT_THRESHOLD = 0.05   # mg/L - prototype cutoff, not a real safety standard
MIN_ACCEPTABLE_PRESSURE = 10.0   # m - prototype assumption, not a real engineering code

SENSOR_LOCATIONS = [
    "J_Ghatkopar", "J_Vidyavihar", "J_Powai",
    "J_Chembur", "J_Sion", "J_MalabarHill",
]
VALVES = ["V1_Kurla_Dadar", "V2_Worli_MalabarHl", "V3_Vikhroli_Andheri", "V4_Kurla_Chembur"]

POPULATION = {
    "J_Bhandup": 2250, "J_Ghatkopar": 1800, "J_Kurla": 1500, "J_Vikhroli": 1200,
    "J_Andheri": 3000, "J_JogeshwariE": 1350, "J_Dadar": 2100, "J_Worli": 1500,
    "J_MalabarHill": 900, "J_Parel": 1650, "J_Sion": 1350, "J_Chembur": 1950,
    "J_Powai": 1050, "J_Vidyavihar": 1200, "J_Matunga": 1500,
}
TOTAL_POPULATION = sum(POPULATION.values())


# ---------------------------------------------------------------------------
# Simulation helpers (cached so repeated UI interactions don't re-solve
# the same scenario over and over)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_template():
    return wntr.network.WaterNetworkModel("aquashield.inp")


@st.cache_data(show_spinner=False)
def run_scenario(closed_links_key):
    """closed_links_key: tuple of link IDs to close (hashable, for caching)."""
    wn = wntr.network.WaterNetworkModel("aquashield.inp")
    for link_id in closed_links_key:
        wn.get_link(link_id).initial_status = "Closed"
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim(file_prefix="app_run")
    return results


def get_node_positions(wn):
    return {name: wn.get_node(name).coordinates for name in wn.node_name_list}


def draw_network(wn, results, color_by, hour, closed_links=()):
    """color_by: 'pressure' or 'quality'. hour: snapshot time in hours."""
    G = nx.Graph()
    pos = get_node_positions(wn)
    G.add_nodes_from(wn.node_name_list)
    edge_status = {}
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        G.add_edge(link.start_node_name, link.end_node_name)
        edge_status[(link.start_node_name, link.end_node_name)] = link_name in closed_links

    t_sec = hour * 3600
    series = results.node[color_by]
    # snap to nearest available timestep
    nearest_t = min(series.index, key=lambda x: abs(x - t_sec))
    values = series.loc[nearest_t]

    fig, ax = plt.subplots(figsize=(9, 7))
    if color_by == "quality":
        node_colors = [values.get(n, 0) for n in G.nodes]
        cmap = "YlOrRd"
        vmin, vmax = 0, max(0.5, max(node_colors))
    else:
        node_colors = [values.get(n, 0) for n in G.nodes]
        cmap = "Blues"
        vmin, vmax = 0, max(node_colors) if node_colors else 1

    edge_colors = ["red" if edge_status.get(e, False) else "gray" for e in G.edges]
    edge_widths = [2.5 if edge_status.get(e, False) else 1.0 for e in G.edges]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths)
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, cmap=cmap, vmin=vmin, vmax=vmax, node_size=450
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    plt.colorbar(nodes, ax=ax, label=("Quality (mg/L)" if color_by == "quality" else "Pressure (m)"))
    ax.set_title(f"Network state at t = {hour:.0f}:00 hrs" + (" (red = closed link)" if closed_links else ""))
    ax.axis("off")
    return fig


def localize_source(results):
    wn = load_template()
    flow = results.link["flowrate"]
    quality = results.node["quality"]
    snapshot_hour = 3.0
    nearest_t = min(flow.index, key=lambda x: abs(x - snapshot_hour * 3600))

    actual_abnormal = {n: quality[n].max() > QUALITY_ALERT_THRESHOLD for n in SENSOR_LOCATIONS}

    G = nx.DiGraph()
    G.add_nodes_from(wn.node_name_list)
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        f = flow.loc[nearest_t, link_name]
        if f >= 0:
            G.add_edge(link.start_node_name, link.end_node_name)
        else:
            G.add_edge(link.end_node_name, link.start_node_name)

    scores = {}
    for cand in wn.junction_name_list:
        reachable = nx.descendants(G, cand) | {cand}
        scores[cand] = sum(1 for n in SENSOR_LOCATIONS if (n in reachable) == actual_abnormal[n])

    max_score = max(scores.values())
    TEMPERATURE = 2.5
    weights = {c: math.exp(TEMPERATURE * (s - max_score)) for c, s in scores.items()}
    wtotal = sum(weights.values())
    confidence = {c: weights[c] / wtotal * 100 for c in scores}

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked, confidence, actual_abnormal


def containment_options(closed_link_combo=None):
    results_list = []
    for combo in itertools.product(["Open", "Closed"], repeat=len(VALVES)):
        closed = tuple(v for v, s in zip(VALVES, combo) if s == "Closed")
        results = run_scenario(closed)
        pressure = results.node["pressure"]
        quality = results.node["quality"]
        exposed_pop = sum(POPULATION[j] for j in POPULATION if quality[j].max() > QUALITY_ALERT_THRESHOLD)
        disrupted_pop = sum(POPULATION[j] for j in POPULATION if pressure[j].min() < MIN_ACCEPTABLE_PRESSURE)
        results_list.append({
            "closed": closed,
            "exposed_pop": exposed_pop,
            "disrupted_pop": disrupted_pop,
            "containment_pct": (1 - exposed_pop / TOTAL_POPULATION) * 100,
            "service_continuity_pct": (TOTAL_POPULATION - disrupted_pop) / TOTAL_POPULATION * 100,
        })
    return results_list


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("AquaShield")
st.caption("Municipal Water Network Digital Twin & Emergency Response System — prototype demo")

wn = load_template()

tab_normal, tab_contam, tab_maint = st.tabs(
    ["Normal Operation", "Inject Contamination", "Maintenance Scenario"]
)

with tab_normal:
    st.subheader("Baseline network status")
    baseline_results = run_scenario(())
    hour = st.slider("Time (hrs)", 0, 24, 3, key="normal_hour")
    fig = draw_network(wn, baseline_results, "pressure", hour)
    st.pyplot(fig)
    st.caption(
        f"Total synthetic demo population represented: {TOTAL_POPULATION:,}. "
        f"Population figures are synthetic (demand-scaled), not real census data."
    )

with tab_contam:
    st.subheader("Contamination event: injected at J_Kurla")
    st.caption(
        "This is a simulated/prototype event, not a real detection. "
        f"Alert threshold: {QUALITY_ALERT_THRESHOLD} mg/L (demo assumption)."
    )
    contam_results = run_scenario(())  # J_Kurla source is baked into the .inp file
    hour = st.slider("Time (hrs)", 0, 24, 3, key="contam_hour")
    fig = draw_network(wn, contam_results, "quality", hour)
    st.pyplot(fig)

    ranked, confidence, actual_abnormal = localize_source(contam_results)
    top_candidate, top_score = ranked[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probable source", top_candidate, f"{confidence[top_candidate]:.1f}% confidence")
        st.write("Sensor readings (ever abnormal during 24h):")
        for n, ab in actual_abnormal.items():
            st.write(f"- {n}: {'🔴 ABNORMAL' if ab else '🟢 normal'}")

    with col2:
        st.write("Top candidate sources (prototype rule-based estimate):")
        for cand, score in ranked[:5]:
            st.write(f"- {cand}: {confidence[cand]:.1f}% (matched {score}/{len(SENSOR_LOCATIONS)} sensors)")

    st.divider()
    st.subheader("Containment options")
    if st.button("Test valve isolation strategies"):
        with st.spinner("Testing all valve combinations..."):
            opts = containment_options()
        seen = set()
        rows = []
        for r in sorted(opts, key=lambda x: x["exposed_pop"]):
            key = (r["exposed_pop"], r["disrupted_pop"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "Close valves": "+".join(r["closed"]) or "none",
                "Population isolated": r["disrupted_pop"],
                "Containment %": round(r["containment_pct"], 1),
                "Service continuity %": round(r["service_continuity_pct"], 1),
            })
        st.table(rows)
        st.caption(
            "These are the distinct trade-offs this valve layout can achieve - "
            "not a single forced 'best' answer. Choose based on which cost "
            "(contamination spread vs. service loss) matters more for this event."
        )

with tab_maint:
    st.subheader("Maintenance: close P10_Vikhroli_Powai")
    before = run_scenario(())
    after = run_scenario(("P10_Vikhroli_Powai",))

    def summarize(results):
        pressure = results.node["pressure"]
        disrupted = [j for j in POPULATION if pressure[j].min() < MIN_ACCEPTABLE_PRESSURE]
        disrupted_pop = sum(POPULATION[j] for j in disrupted)
        return {
            "min_pressure_powai": pressure["J_Powai"].min(),
            "disrupted_zones": disrupted,
            "disrupted_pop": disrupted_pop,
            "service_continuity_pct": (TOTAL_POPULATION - disrupted_pop) / TOTAL_POPULATION * 100,
        }

    b, a = summarize(before), summarize(after)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**BEFORE (P10 open)**")
        st.metric("Min pressure @ Powai", f"{b['min_pressure_powai']:.1f} m")
        st.metric("Service continuity", f"{b['service_continuity_pct']:.1f}%")
    with col2:
        st.write("**AFTER (P10 closed for maintenance)**")
        st.metric("Min pressure @ Powai", f"{a['min_pressure_powai']:.1f} m",
                   delta=f"{a['min_pressure_powai'] - b['min_pressure_powai']:.1f} m")
        st.metric("Service continuity", f"{a['service_continuity_pct']:.1f}%")

    hour = st.slider("Time (hrs)", 0, 24, 3, key="maint_hour")
    fig = draw_network(wn, after, "pressure", hour, closed_links=("P10_Vikhroli_Powai",))
    st.pyplot(fig)

    if a["disrupted_zones"]:
        st.warning(f"Zones below {MIN_ACCEPTABLE_PRESSURE}m: {', '.join(a['disrupted_zones'])}")
    else:
        st.success(
            "No zone drops below the minimum pressure threshold - the existing loop route "
            "(Vikhroli → Andheri → Jogeshwari E → Powai) covers the maintenance window safely."
        )
