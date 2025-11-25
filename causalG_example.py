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
- Appends a LaTeX snippet to the uploaded TeX at "/mnt/data/main (27).tex" (if present)
- Tune parameters below (MAX_STEPS) to control enumeration depth

Requirements: networkx, matplotlib, (optional pandas)
"""

import os
import json
from collections import defaultdict, namedtuple
import networkx as nx
import matplotlib.pyplot as plt
import math

# ---------- PARAMETERS ----------
OUT_DIR = "/mnt/data/multiway_outputs"   # change if desired
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

    for step in range(1, max_steps+1):
        new_frontier = []
        for state in frontier:
            applicable = system.applicable(state)
            if not applicable:
                continue
            paths_here = paths_per_state.get(state, [{"events": [], "J": 0.0, "last_event": ROOT_EVENT}])
            for path_meta in paths_here:
                for rule in applicable:
                    ev_id = f"E{event_counter}"; event_counter += 1
                    j = rule.J()
                    ev = Event(id=ev_id, rule_name=rule.name, src_state=state, dst_state=rule.dst,
                               j=j, effort=rule.effort, utility=rule.utility, constraints=rule.constraints, step=step)
                    events[ev_id] = ev
                    # causal edges from creators of tokens in state -> this event
                    creators = token_creators.get(state, [ROOT_EVENT])
                    for cr in creators:
                        causal.add_edge(cr, ev_id)
                    causal.add_node(ev_id, label=f"{ev_id}\\n{rule.name}\\nJ={j:.2f}")
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
        frontier = list(dict.fromkeys(new_frontier))
    # flatten paths
    all_paths = []
    for s, plist in paths_per_state.items():
        for p in plist:
            all_paths.append({"state": s, "events": p["events"], "J": p["J"]})
    return multiway, causal, events, all_paths

# Drawing helpers (use default matplotlib settings; do not set custom colors)
def draw_multiway(G, outpath):
    plt.figure(figsize=(6,4))
    pos = nx.spring_layout(G, seed=2)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, font_size=10)
    edge_labels = {(u,v): d.get('event','') for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Multiway state graph")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def draw_causal(C, outpath):
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(C, seed=3)
    labels = {n: d.get('label', str(n)) for n,d in C.nodes(data=True)}
    nx.draw(C, pos, with_labels=False, node_size=600)
    nx.draw_networkx_labels(C, pos, labels=labels, font_size=8)
    plt.title("Causal graph (events as nodes)")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_hist(Js, outpath):
    plt.figure(figsize=(6,4))
    plt.hist(Js, bins=12)
    plt.xlabel("Path J value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def main():
    glass = make_glass_system()
    plant = make_plant_system()

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

    print("Drawing graphs (may take a moment)...")
    draw_multiway(g_multi, g_multi_png)
    draw_multiway(p_multi, p_multi_png)
    draw_causal(g_causal, g_causal_png)
    draw_causal(p_causal, p_causal_png)

    # Save events & paths
    save_json({k: v._asdict() for k,v in g_events.items()}, os.path.join(OUT_DIR, "glass_events.json"))
    save_json({k: v._asdict() for k,v in p_events.items()}, os.path.join(OUT_DIR, "plant_events.json"))
    save_json(g_paths, os.path.join(OUT_DIR, "glass_paths.json"))
    save_json(p_paths, os.path.join(OUT_DIR, "plant_paths.json"))

    # Plot histograms
    plot_hist([p["J"] for p in g_paths], g_hist)
    plot_hist([p["J"] for p in p_paths], p_hist)

    # Append LaTeX snippet to uploaded tex
    uploaded_tex = "/mnt/data/main (27).tex"
    updated_tex = "/mnt/data/main_updated_with_causal.tex"
    latex_snippet = r"""
% ---- Inserted causal-graph decision-theoretic section ----
\section{Causal-graph construction and decision-theoretic action functional}
We implement a multiway causal-graph construction where each discrete rule application
is recorded as an event node and directed edges point from events that created tokens
to events that later consume them. Each event $e$ is annotated with a per-event action
functional
\begin{equation}
j(e) \;=\; \mathrm{Effort}(e) \;-\; \mathrm{Utility}(e) \;+\; \mathrm{Constraints}(e),
\end{equation}
and any trajectory $\gamma=(e_1,\dots,e_n)$ accumulates $J[\gamma]=\sum_t j(e_t)$.
We simulate two rule-sets on the shared state alphabet $\{I,D,E\}$ (intact, damaged, end)
to illustrate fragility (glass) and resilience (plant). Figures and code used to
generate the causal graphs are included in the project repository.
% ---- end snippet ----
"""
    if os.path.exists(uploaded_tex):
        with open(uploaded_tex, "r", encoding="utf-8") as f:
            original = f.read()
        with open(updated_tex, "w", encoding="utf-8") as f:
            f.write(original)
            f.write("\n\n% ---- Added by assistant ----\n")
            f.write(latex_snippet)
        print("Wrote updated TeX to:", updated_tex)
    else:
        with open(updated_tex, "w", encoding="utf-8") as f:
            f.write("% Created by assistant - causal graph snippet\n")
            f.write(latex_snippet)
        print("Created new TeX with snippet at:", updated_tex)

    # Print produced files
    produced = {
        "glass_multiway": g_multi_png,
        "plant_multiway": p_multi_png,
        "glass_causal": g_causal_png,
        "plant_causal": p_causal_png,
        "glass_events_json": os.path.join(OUT_DIR, "glass_events.json"),
        "plant_events_json": os.path.join(OUT_DIR, "plant_events.json"),
        "glass_paths_json": os.path.join(OUT_DIR, "glass_paths.json"),
        "plant_paths_json": os.path.join(OUT_DIR, "plant_paths.json"),
        "glass_hist": g_hist,
        "plant_hist": p_hist,
        "updated_tex": updated_tex
    }
    print("Produced files (saved under OUT_DIR):")
    for k,v in produced.items():
        print(" -", k, ":", v)

if __name__ == "__main__":
    main()
