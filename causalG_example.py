#!/usr/bin/env python3
"""
multiway_causal.py

- Implements multiway (single-application branching) enumerator over the shared state alphabet {I,D,E}
- Builds per-event action functional j = effort - utility + constraints
- Builds:
    * multiway state graph (states as nodes, edges labeled by event id)
    * causal graph (events as nodes, edges from creator events -> event)
    * JSON manifests of events and enumerated paths
    * visualizations (PNG) for multiway and causal graphs and histograms of path J
    * animated GIFs showing the evolution of the multiway graph by depth
- Tune parameters below (MAX_STEPS) to control enumeration depth

Requirements: networkx, matplotlib, imageio (for GIF generation)
"""

import os
import json
from collections import defaultdict, namedtuple
import networkx as nx
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import imageio.v2 as imageio
import tempfile

# ---------- PARAMETERS ----------
OUT_DIR = "multiway_outputs"   # change if desired
os.makedirs(OUT_DIR, exist_ok=True)
MAX_STEPS = 5       # depth of multiway expansion; increase carefully (exponential growth)
INITIAL_STATE = "I"
ROOT_EVENT = "ROOT_EV"
# --------------------------------

Event = namedtuple("Event", ["id", "rule_name", "src_state", "dst_state", "j", "effort", "utility", "constraints", "step"])

class Transition:
    def __init__(self, name, src, dst, effort, utility, constraints):
        self.name = name
        self.src = src
        self.dst = dst
        self.effort = float(effort)
        self.utility = float(utility)
        self.constraints = float(constraints)
    def J(self):
        return self.effort - self.utility + self.constraints
    def to_dict(self):
        return {"name": self.name, "src": self.src, "dst": self.dst, "effort": self.effort,
                "utility": self.utility, "constraints": self.constraints, "J": self.J()}

class System:
    def __init__(self, transitions):
        self.rules = defaultdict(list)
        for t in transitions:
            self.rules[t.src].append(t)
    def applicable(self, state):
        return self.rules[state] if state in self.rules else []

def make_glass_system():
    T = []
    # I -> {I, D, E}
    T.append(Transition("stay_intact", "I", "I", effort=0.1, utility=0.0, constraints=0.0))
    T.append(Transition("damage",      "I", "D", effort=0.4, utility=0.0, constraints=0.0))
    T.append(Transition("catastrophe", "I", "E", effort=0.2, utility=0.0, constraints=1.0))
    # D -> {D, E}
    T.append(Transition("remain_damaged", "D", "D", effort=0.3, utility=0.0, constraints=0.0))
    T.append(Transition("fail",           "D", "E", effort=0.1, utility=0.0, constraints=1.0))
    return System(T)

def make_plant_system():
    T = []
    # I -> {I, D}
    T.append(Transition("stay_intact", "I", "I", effort=0.1, utility=0.0, constraints=0.0))
    T.append(Transition("injure",      "I", "D", effort=0.4, utility=0.0, constraints=0.0))
    # D -> {I, D, E}
    T.append(Transition("heal",  "D", "I", effort=0.3, utility=0.6, constraints=0.0))
    T.append(Transition("linger","D", "D", effort=0.2, utility=0.0, constraints=0.0))
    T.append(Transition("die",   "D", "E", effort=0.2, utility=0.0, constraints=1.0))
    return System(T)

def enumerate_multiway(system, initial_state="I", max_steps=4):
    """
    Returns:
      - multiway graph (state nodes) with edges labeled by event id
      - causal graph (event nodes) with causal edges
      - events dict: event_id -> Event
      - all_paths: list of dicts {state, events(list), J}
    """
    multiway = nx.DiGraph()
    causal = nx.DiGraph()
    events = {}
    state_seen = {initial_state: initial_state}
    multiway.add_node(initial_state, label=initial_state, depth=0)
    token_creators = { initial_state: [ROOT_EVENT] }
    causal.add_node(ROOT_EVENT, label=ROOT_EVENT)
    frontier = [initial_state]
    event_counter = 0
    paths_per_state = { initial_state: [ {"events": [], "J": 0.0, "last_event": ROOT_EVENT} ] }

    for step in tqdm(range(1, max_steps+1), desc="Enumerating steps", unit="step"):
        new_frontier = []
        for state in frontier:
            applicable = system.applicable(state)
            if not applicable:
                continue
            paths_here = list(paths_per_state.get(state, [{"events": [], "J": 0.0, "last_event": ROOT_EVENT}]))
            # The real work happens here - nested loops over paths and rules
            total_iterations = len(paths_here) * len(applicable)
            pbar = tqdm(total=total_iterations, desc=f"  Step {step}: state '{state}'", leave=False, unit="event")
            for path_meta in paths_here:
                for rule in applicable:
                    pbar.update(1)
                    ev_id = f"E{event_counter}"; event_counter += 1
                    j = rule.J()
                    ev = Event(id=ev_id, rule_name=rule.name, src_state=state, dst_state=rule.dst,
                               j=j, effort=rule.effort, utility=rule.utility, constraints=rule.constraints, step=step)
                    events[ev_id] = ev
                    # causal edges from creators of tokens in state -> this event
                    creators = token_creators.get(state, [ROOT_EVENT])
                    for cr in creators:
                        causal.add_edge(cr, ev_id)
                    causal.add_node(ev_id, label=f"{ev_id}\\n{rule.name}\\nJ={j:.2f}", step=step)
                    # multiway edge state -> new_state labeled by ev
                    new_state = rule.dst
                    multiway.add_node(new_state, label=new_state, depth=step)
                    multiway.add_edge(state, new_state, event=ev_id, rule=rule.name, j=j)
                    # new state's creator token is this event (single-symbol model)
                    token_creators[new_state] = [ev_id]
                    # extend paths
                    new_paths = paths_per_state.get(new_state, [])
                    new_path = {"events": path_meta["events"] + [ev_id], "J": path_meta["J"] + j, "last_event": ev_id}
                    new_paths.append(new_path)
                    paths_per_state[new_state] = new_paths
                    new_frontier.append(new_state)
            pbar.close()
        frontier = list(dict.fromkeys(new_frontier))
        # Print stats after each step to show growth
        total_paths = sum(len(plist) for plist in paths_per_state.values())
        tqdm.write(f"    After step {step}: {len(events)} events, {len(frontier)} unique states in frontier, {total_paths} total paths")
    # flatten paths
    all_paths = []
    for s, plist in paths_per_state.items():
        for p in plist:
            all_paths.append({"state": s, "events": p["events"], "J": p["J"]})
    return multiway, causal, events, all_paths

# Drawing helpers
def hierarchical_layout(G, width=3.0, height=2.0):
    """Create hierarchical layout based on node depth attribute (cone of growth)"""
    pos = {}
    depth_nodes = defaultdict(list)

    # Group nodes by depth
    for node, data in G.nodes(data=True):
        depth = data.get('depth', 0)
        depth_nodes[depth].append(node)

    max_depth = max(depth_nodes.keys()) if depth_nodes else 0

    # Position nodes: depth determines y, spread horizontally within depth level
    for depth, nodes in depth_nodes.items():
        y = height * (1 - depth / max(max_depth, 1))  # Top to bottom
        n = len(nodes)
        for i, node in enumerate(nodes):
            # Spread nodes horizontally, centered
            if n == 1:
                x = 0
            else:
                x = width * (i / (n - 1) - 0.5)
            pos[node] = (x, y)

    return pos

def draw_multiway(G, outpath):
    if len(G.nodes()) > 1000:
        print(f"Skipping drawing {outpath} - graph too large ({len(G.nodes())} nodes)")
        return

    plt.figure(figsize=(12, 8))
    pos = hierarchical_layout(G, width=10, height=6)
    labels = nx.get_node_attributes(G, 'label')

    # Color nodes by depth
    node_colors = [G.nodes[n].get('depth', 0) for n in G.nodes()]
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=700,
            font_size=10, node_color=node_colors, cmap='viridis')

    edge_labels = {(u,v): d.get('event','') for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Multiway state graph (hierarchical by depth)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def draw_causal(C, outpath):
    if len(C.nodes()) > 5000:
        print(f"Skipping drawing {outpath} - graph too large ({len(C.nodes())} nodes)")
        return

    plt.figure(figsize=(12, 8))
    # For causal graph, use step attribute for hierarchical layout
    pos = {}
    step_nodes = defaultdict(list)
    for node, data in C.nodes(data=True):
        # Extract step from label or use 0 for ROOT
        if node == ROOT_EVENT:
            step = 0
        else:
            step = data.get('step', 0)
        step_nodes[step].append(node)

    max_step = max(step_nodes.keys()) if step_nodes else 0
    for step, nodes in step_nodes.items():
        y = 6 * (1 - step / max(max_step, 1))
        n = len(nodes)
        for i, node in enumerate(nodes):
            if n == 1:
                x = 0
            else:
                x = 10 * (i / (n - 1) - 0.5)
            pos[node] = (x, y)

    labels = {n: d.get('label', str(n)) for n, d in C.nodes(data=True)}
    node_colors = [step_nodes[0].index(n) if n in step_nodes[0] else
                   next((s for s, nodes in step_nodes.items() if n in nodes), 0)
                   for n in C.nodes()]

    nx.draw(C, pos, with_labels=False, node_size=400, node_color=node_colors, cmap='plasma')
    nx.draw_networkx_labels(C, pos, labels=labels, font_size=6)
    plt.title("Causal graph (events as nodes, hierarchical by step)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_hist(Js, outpath):
    plt.figure(figsize=(6,4))
    plt.hist(Js, bins=12)
    plt.xlabel("Path J value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def create_network_growth_gif(graph, max_steps, outpath, title="Network Growth", graph_type="multiway"):
    """Create animated GIF showing network growing step by step"""
    if len(graph.nodes()) > 5000:
        print(f"Skipping GIF {outpath} - graph too large ({len(graph.nodes())} nodes)")
        return

    frames = []
    temp_dir = tempfile.mkdtemp()

    # Use hierarchical layout for all frames (consistent positioning)
    full_pos = hierarchical_layout(graph, width=10, height=6)

    for step in range(max_steps + 1):
        # Filter nodes up to current step/depth
        if graph_type == "multiway":
            nodes_at_step = [n for n, d in graph.nodes(data=True) if d.get('depth', 0) <= step]
        else:  # causal
            nodes_at_step = [n for n, d in graph.nodes(data=True)
                           if d.get('step', 0) <= step or n == ROOT_EVENT]

        subgraph = graph.subgraph(nodes_at_step).copy()
        if len(subgraph.nodes()) == 0:
            continue

        # Use positions from full graph for consistency
        pos = {n: full_pos[n] for n in subgraph.nodes() if n in full_pos}

        # Draw frame
        fig, ax = plt.subplots(figsize=(12, 8))
        labels = nx.get_node_attributes(subgraph, 'label')

        # Color nodes by step/depth
        if graph_type == "multiway":
            node_colors = [graph.nodes[n].get('depth', 0) for n in subgraph.nodes()]
            cmap = 'viridis'
        else:
            node_colors = [graph.nodes[n].get('step', 0) if n != ROOT_EVENT else 0
                          for n in subgraph.nodes()]
            cmap = 'plasma'

        nx.draw(subgraph, pos, with_labels=True, labels=labels,
                node_size=500, font_size=8, node_color=node_colors,
                cmap=cmap, vmin=0, vmax=max_steps, ax=ax)

        if graph_type == "multiway" and len(subgraph.edges()) < 100:
            edge_labels = {(u,v): d.get('event','')[:5] for u,v,d in subgraph.edges(data=True)}
            nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=6, ax=ax)

        ax.set_title(f"{title} - Step {step}/{max_steps} ({len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges)")
        plt.tight_layout()

        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{step:03d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()

        frames.append(imageio.imread(frame_path))

    if not frames:
        print(f"No frames generated for {outpath}")
        return

    # Create GIF with pause on last frame
    final_frames = frames + [frames[-1]] * 3  # Hold last frame
    imageio.mimsave(outpath, final_frames, duration=0.5, loop=0)

    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir)

    print(f"Created network growth GIF: {outpath}")

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def export_causal_for_gephi(causal_graph, events, outpath):
    """Export causal graph in GEXF format for Gephi"""
    try:
        # Add event data as node attributes
        G = causal_graph.copy()
        for event_id, event_data in events.items():
            if event_id in G.nodes:
                G.nodes[event_id]['rule_name'] = event_data.rule_name
                G.nodes[event_id]['src_state'] = event_data.src_state
                G.nodes[event_id]['dst_state'] = event_data.dst_state
                G.nodes[event_id]['j'] = event_data.j
                G.nodes[event_id]['effort'] = event_data.effort
                G.nodes[event_id]['utility'] = event_data.utility
                G.nodes[event_id]['constraints'] = event_data.constraints

        nx.write_gexf(G, outpath)
        print(f"Exported GEXF for Gephi: {outpath}")
    except Exception as e:
        print(f"Error exporting GEXF: {e}")

def export_causal_for_dynetvis(causal_graph, events, outpath):
    """Export causal graph in temporal JSON format for DyNetVis"""
    # DyNetVis format: nodes with appearance/disappearance times, edges with timestamps
    nodes = []
    edges = []

    # Create nodes with temporal information
    for node_id, node_data in causal_graph.nodes(data=True):
        if node_id == ROOT_EVENT:
            step = 0
        else:
            event = events.get(node_id)
            step = event.step if event else node_data.get('step', 0)

        node_entry = {
            "id": node_id,
            "label": node_id,
            "start": step,
            "end": step + 1  # Event exists from its creation step onwards
        }

        # Add event attributes
        if node_id != ROOT_EVENT and node_id in events:
            event = events[node_id]
            node_entry.update({
                "rule": event.rule_name,
                "src_state": event.src_state,
                "dst_state": event.dst_state,
                "j": event.j,
                "effort": event.effort,
                "utility": event.utility,
                "constraints": event.constraints
            })

        nodes.append(node_entry)

    # Create edges with timestamps
    for source, target, edge_data in causal_graph.edges(data=True):
        # Edge appears when target event is created
        target_step = events[target].step if target in events else 1
        edge_entry = {
            "source": source,
            "target": target,
            "start": target_step,
            "end": target_step + 1
        }
        edges.append(edge_entry)

    output = {
        "nodes": nodes,
        "edges": edges,
        "directed": True,
        "metadata": {
            "description": "Causal graph showing event dependencies",
            "max_step": max(events[e].step for e in events) if events else 0
        }
    }

    save_json(output, outpath)
    print(f"Exported temporal JSON for DyNetVis: {outpath}")

def main():
    glass = make_glass_system()
    plant = make_plant_system()

    if MAX_STEPS > 5:
        print(f"WARNING: MAX_STEPS={MAX_STEPS} may cause exponential memory growth!")
        print("Consider starting with MAX_STEPS=2-3 to test performance.")

    print("Enumerating glass multiway up to", MAX_STEPS, "steps ...")
    g_multi, g_causal, g_events, g_paths = enumerate_multiway(glass, initial_state=INITIAL_STATE, max_steps=MAX_STEPS)
    print(f"glass: events={len(g_events)}, paths={len(g_paths)}")

    print("Enumerating plant multiway up to", MAX_STEPS, "steps ...")
    p_multi, p_causal, p_events, p_paths = enumerate_multiway(plant, initial_state=INITIAL_STATE, max_steps=MAX_STEPS)
    print(f"plant: events={len(p_events)}, paths={len(p_paths)}")

    # Save images
    g_multi_png = os.path.join(OUT_DIR, "glass_multiway.png")
    p_multi_png = os.path.join(OUT_DIR, "plant_multiway.png")
    g_causal_png = os.path.join(OUT_DIR, "glass_causal.png")
    p_causal_png = os.path.join(OUT_DIR, "plant_causal.png")
    g_hist = os.path.join(OUT_DIR, "glass_J_hist.png")
    p_hist = os.path.join(OUT_DIR, "plant_J_hist.png")
    g_multi_gif = os.path.join(OUT_DIR, "glass_multiway_growth.gif")
    p_multi_gif = os.path.join(OUT_DIR, "plant_multiway_growth.gif")

    print("Drawing graphs (may take a moment)...")
    graphs_to_draw = [
        ("Glass multiway", lambda: draw_multiway(g_multi, g_multi_png)),
        ("Plant multiway", lambda: draw_multiway(p_multi, p_multi_png)),
        ("Glass causal", lambda: draw_causal(g_causal, g_causal_png)),
        ("Plant causal", lambda: draw_causal(p_causal, p_causal_png))
    ]
    for desc, draw_func in tqdm(graphs_to_draw, desc="Drawing graphs", unit="graph"):
        draw_func()

    # Save events & paths
    print("Saving JSON files...")
    json_files = [
        ("Glass events", lambda: save_json({k: v._asdict() for k,v in g_events.items()}, os.path.join(OUT_DIR, "glass_events.json"))),
        ("Plant events", lambda: save_json({k: v._asdict() for k,v in p_events.items()}, os.path.join(OUT_DIR, "plant_events.json"))),
        ("Glass paths", lambda: save_json(g_paths, os.path.join(OUT_DIR, "glass_paths.json"))),
        ("Plant paths", lambda: save_json(p_paths, os.path.join(OUT_DIR, "plant_paths.json")))
    ]
    for desc, save_func in tqdm(json_files, desc="Saving JSON", unit="file"):
        save_func()

    # Export causal graphs for external tools
    print("Exporting causal graphs for Gephi and DyNetVis...")
    g_gephi = os.path.join(OUT_DIR, "glass_causal.gexf")
    p_gephi = os.path.join(OUT_DIR, "plant_causal.gexf")
    g_dynetvis = os.path.join(OUT_DIR, "glass_causal_temporal.json")
    p_dynetvis = os.path.join(OUT_DIR, "plant_causal_temporal.json")

    export_files = [
        ("Glass Gephi export", lambda: export_causal_for_gephi(g_causal, g_events, g_gephi)),
        ("Plant Gephi export", lambda: export_causal_for_gephi(p_causal, p_events, p_gephi)),
        ("Glass DyNetVis export", lambda: export_causal_for_dynetvis(g_causal, g_events, g_dynetvis)),
        ("Plant DyNetVis export", lambda: export_causal_for_dynetvis(p_causal, p_events, p_dynetvis))
    ]
    for desc, export_func in tqdm(export_files, desc="Exporting graphs", unit="file"):
        export_func()

    # Plot histograms
    print("Plotting histograms...")
    histograms = [
        ("Glass J histogram", lambda: plot_hist([p["J"] for p in g_paths], g_hist)),
        ("Plant J histogram", lambda: plot_hist([p["J"] for p in p_paths], p_hist))
    ]
    for desc, plot_func in tqdm(histograms, desc="Plotting histograms", unit="plot"):
        plot_func()

    # Create network growth GIFs
    print("Creating network growth GIFs...")
    gifs_to_create = [
        ("Glass multiway GIF", lambda: create_network_growth_gif(g_multi, MAX_STEPS, g_multi_gif, "Glass Multiway Growth", "multiway")),
        ("Plant multiway GIF", lambda: create_network_growth_gif(p_multi, MAX_STEPS, p_multi_gif, "Plant Multiway Growth", "multiway"))
    ]
    for desc, gif_func in tqdm(gifs_to_create, desc="Creating GIFs", unit="gif"):
        gif_func()

    # Print produced files
    produced = {
        "glass_multiway": g_multi_png,
        "plant_multiway": p_multi_png,
        "glass_causal": g_causal_png,
        "plant_causal": p_causal_png,
        "glass_causal_gephi": g_gephi,
        "plant_causal_gephi": p_gephi,
        "glass_causal_dynetvis": g_dynetvis,
        "plant_causal_dynetvis": p_dynetvis,
        "glass_events_json": os.path.join(OUT_DIR, "glass_events.json"),
        "plant_events_json": os.path.join(OUT_DIR, "plant_events.json"),
        "glass_paths_json": os.path.join(OUT_DIR, "glass_paths.json"),
        "plant_paths_json": os.path.join(OUT_DIR, "plant_paths.json"),
        "glass_hist": g_hist,
        "plant_hist": p_hist,
        "glass_multiway_growth_gif": g_multi_gif,
        "plant_multiway_growth_gif": p_multi_gif
    }
    print("Produced files (saved under OUT_DIR):")
    for k,v in produced.items():
        print(" -", k, ":", v)

if __name__ == "__main__":
    main()
