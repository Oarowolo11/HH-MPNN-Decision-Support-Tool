"""
precompute_year.py
==================
One-off batch job: run the trained HH-MPNN over all 35040 yearly load
scenarios (one per 15 minutes) for the 118-bus system, evaluate line-flow and
power-balance violations plus dispatch cost for every scenario, and save
everything the Streamlit app needs into a single results file.

Run once (GPU strongly recommended):
    python precompute_year.py

The pipeline mirrors the notebook cell-by-cell:
  - static grid data + topology  : cells 8-9, 13-15, 34-36
  - positional encodings         : cells 16-29
  - HeteroData / DataLoader      : cells 31, 42-44
  - inference with bound scaling : cell 54 (test_model)
  - branch flows & violations    : cells 71-84, 100-109
  - power balance mismatches     : cells 110-112
  - dispatch cost                : cell 85 (compute_optimality, cost part only)
"""

import numpy as np
import torch
from torch.amp import autocast
from torch_geometric.loader import DataLoader
import networkx as nx

from inference_core import (
    HeteroInteractGNN,
    create_grid_hetero_data,
    convert_voltage_bounds,
    convert_power_bounds,
    calculate_only_branch_flows,
    convert_to_complex_voltage,
    convert_to_complex_rectangle,
    convert_to_power_magnitude,
    compute_gandb,
    get_B_matrix,
    adjacency_to_laplacian,
    effective_resistance_matrix,
    compute_row_statistics_vectorized,
)

# ---------------------------------------------------------------------------
# Configuration — adjust the paths to your environment
# ---------------------------------------------------------------------------
system_size = 118

# Static grid data (topology + constant features), same file as the notebook
OPFDATA_PATH = f'/path_to/{system_size}bus_combined_dataset.npz'

# Yearly load scenarios: expects an .npz with key 'grid_load' of shape
# (35040, n_load, 2) [Pd, Qd] in per unit, one snapshot per 15 minutes.
# ASSUMPTION: same load ordering and units as the PGLearn 'grid_load' arrays.
YEAR_LOADS_PATH = f'./data/{system_size}_year_loads.npz'

# Trained weights (saved in notebook cell 61)
WEIGHTS_PATH = f'./{system_size}_bus_HybridHeteroGNN_5_256_PQVT_PGLearn.pth'

# Output file consumed by the Streamlit app
RESULTS_PATH = f'./results_{system_size}_year.npz'

batch_size = 256      # inference chunk size (memory/speed trade-off)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ---------------------------------------------------------------------------
# 1. Load static grid data (notebook cells 8-9, inference-relevant parts)
# ---------------------------------------------------------------------------
data = np.load(OPFDATA_PATH)

grid_bus = data['grid_bus']
grid_generator = data['grid_generator']
grid_shunt = data['grid_shunt']
grid_ac_line_features = data['grid_ac_line_features']
grid_transformer_features = data['grid_transformer_features']
grid_ac_line_receivers = data['grid_ac_line_receivers']
grid_ac_line_senders = data['grid_ac_line_senders']
grid_transformer_senders = data['grid_transformer_senders']
grid_transformer_receivers = data['grid_transformer_receivers']

grid_generator_link_receivers = data['grid_generator_link_receivers']
grid_load_link_receivers = data['grid_load_link_receivers']
grid_shunt_link_receivers = data['grid_shunt_link_receivers']

# Topology is identical across samples -> take sample 0 (as in the notebook)
generator_indices = grid_generator_link_receivers[0]
load_indices = grid_load_link_receivers[0]
shunt_indices = grid_shunt_link_receivers[0]
grid_transformer_senders = grid_transformer_senders[0]
grid_transformer_receivers = grid_transformer_receivers[0]
grid_ac_line_senders = grid_ac_line_senders[0]
grid_ac_line_receivers = grid_ac_line_receivers[0]

# ---------------------------------------------------------------------------
# 2. Yearly load scenarios (the only time-varying input)
# ---------------------------------------------------------------------------
year_data = np.load(YEAR_LOADS_PATH)
grid_load = year_data['grid_load']          # (35040, n_load, 2)

# The model expects one load node per PGLearn load (99 for the 118-bus
# system), not one per bus. If the yearly file stores loads per-bus
# (shape (T, 118, 2) with zeros at non-load buses), extract only the
# entries at actual load buses, in the load-node ordering given by
# load_indices (load node k sits at bus load_indices[k]).
if grid_load.shape[1] == system_size:
    grid_load = grid_load[:, load_indices, :]

# Hard guard: anything else means the load ordering doesn't match training
assert grid_load.shape[1] == len(load_indices), (
    f"grid_load has {grid_load.shape[1]} loads per scenario, "
    f"expected {len(load_indices)} (PGLearn load-node convention)"
)

n_scenarios = grid_load.shape[0]
print(f'Loaded {n_scenarios} load scenarios of shape {grid_load.shape[1:]}')

# ---------------------------------------------------------------------------
# 3. Branch list and unified edge inputs (notebook cells 13-15)
# ---------------------------------------------------------------------------
grid_ac_line_senders = grid_ac_line_senders.reshape(-1, 1)
grid_ac_line_receivers = grid_ac_line_receivers.reshape(-1, 1)
grid_transformer_senders = grid_transformer_senders.reshape(-1, 1)
grid_transformer_receivers = grid_transformer_receivers.reshape(-1, 1)

# AC lines first, transformers appended after (order matters downstream)
branch_list = list(zip(grid_ac_line_senders.flatten(), grid_ac_line_receivers.flatten()))
transformer_list = list(zip(grid_transformer_senders.flatten(), grid_transformer_receivers.flatten()))
for k in transformer_list:
    branch_list.append(k)

# Rearranging edge inputs to align columns between lines and transformers
edge_inputs = np.zeros((len(branch_list), 11))
edge_inputs[:grid_ac_line_features.shape[1], :9] = grid_ac_line_features[0]
edge_inputs[grid_ac_line_features.shape[1]:, :2] = grid_transformer_features[0, :, :2]
edge_inputs[grid_ac_line_features.shape[1]:, 2:4] = grid_transformer_features[0, :, 9:]
edge_inputs[grid_ac_line_features.shape[1]:, 4:9] = grid_transformer_features[0, :, 2:7]
edge_inputs[grid_ac_line_features.shape[1]:, 9:] = grid_transformer_features[0, :, 7:9]
edge_inputs[:grid_ac_line_features.shape[1], 9:10] = 1.0  # tap ratio = 1 for plain lines

# ---------------------------------------------------------------------------
# 4. Positional encodings from effective resistance (notebook cells 16-29)
#    Topology is fixed for this app, so this is computed exactly once.
# ---------------------------------------------------------------------------
edge_g, edge_b = compute_gandb(edge_inputs)

B_weighted = get_B_matrix(system_size, branch_list, torch.tensor(edge_b))
b_mat = np.array(B_weighted)
B_lap = adjacency_to_laplacian(b_mat)
e_R = effective_resistance_matrix(B_lap)
raw_PE = compute_row_statistics_vectorized(e_R)
bus_pe = torch.tensor(raw_PE, dtype=torch.float)

# ---------------------------------------------------------------------------
# 5. Constant per-scenario features (notebook cells 34-36):
#    every feature except the loads is frozen at sample 0.
# ---------------------------------------------------------------------------
samp_grid_bus = grid_bus[0]              # (118, 4)
samp_grid_generator = grid_generator[0]  # (n_gen, 11)
samp_grid_shunt = grid_shunt[0]          # (n_shunt, 2)
samp_ac_line = grid_ac_line_features[0]
samp_transformer = grid_transformer_features[0]

n_gen = samp_grid_generator.shape[0]
n_branch = len(branch_list)

# ---------------------------------------------------------------------------
# 6. Admittance quantities for the flow physics (notebook cells 80, 83)
# ---------------------------------------------------------------------------
edge_inputs_t = torch.tensor(edge_inputs)

conductance_susceptance = np.concatenate((edge_g, edge_b), axis=1)
conductance_susceptance = torch.tensor(conductance_susceptance).to(torch.float32)

charging_susceptance = torch.zeros_like(conductance_susceptance).to(torch.float32)
charging_susceptance[:, 1:] = edge_inputs_t[:, 2:3].to(torch.float32)

Tij = edge_inputs_t[:, 9:].to(torch.float32)
Tij[:grid_ac_line_features.shape[1], 0:1] = 1.0
Tij_rec = convert_to_complex_rectangle(Tij)

# Shunt admittance scattered onto buses; constant across scenarios, so a
# single (1, 118, 2) tensor is enough (calculate_only_branch_flows expands it)
Yks = torch.zeros(1, system_size, samp_grid_shunt.shape[-1]).to(torch.float32)
Yks[:, shunt_indices, :] = torch.tensor(samp_grid_shunt, dtype=torch.float32)
Yks = Yks[:, :, [1, 0]]  # reorder (bs, gs) -> (gs, bs) as in notebook cell 83

# Branch flow limit = long-term line rating (notebook cell 108)
long_term_line_rating = edge_inputs_t[:, 6:7].to(torch.float32)  # (n_branch, 1)

# Cost coefficients from generator features (notebook cell 85)
c2 = torch.tensor(samp_grid_generator[:, 8:9], dtype=torch.float32)   # (n_gen, 1)
c1 = torch.tensor(samp_grid_generator[:, 9:10], dtype=torch.float32)
c0 = torch.tensor(samp_grid_generator[:, 10:11], dtype=torch.float32)

# Forward/reverse key ordering for extracting flows (notebook cells 100-105)
forward_keys = [(i, j, index) for index, (i, j) in enumerate(branch_list)]
reverse_keys = [(j, i, index) for index, (i, j) in enumerate(branch_list)]

# ---------------------------------------------------------------------------
# 7. Model
# ---------------------------------------------------------------------------
model = HeteroInteractGNN().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()
print('Model weights loaded.')

# ---------------------------------------------------------------------------
# 8. Chunked inference + physics over all scenarios
#    (create_dataloader logic from cell 31, applied chunk-wise to keep the
#     dataset list from holding 35040 HeteroData objects at once)
# ---------------------------------------------------------------------------

# Result accumulators (float32 to keep the results file compact)
all_p_pred = np.zeros((n_scenarios, n_gen, 2), dtype=np.float32)          # [Pg, Qg]
all_v_pred = np.zeros((n_scenarios, system_size, 2), dtype=np.float32)    # [Va, Vm]
all_pbal_p = np.zeros((n_scenarios, system_size), dtype=np.float32)       # active balance mismatch
all_pbal_q = np.zeros((n_scenarios, system_size), dtype=np.float32)       # reactive balance mismatch
all_fwd_viol = np.zeros((n_scenarios, n_branch), dtype=np.float32)        # forward flow violation
all_rev_viol = np.zeros((n_scenarios, n_branch), dtype=np.float32)        # reverse flow violation
all_fwd_flow = np.zeros((n_scenarios, n_branch), dtype=np.float32)        # forward |S| (for loading display)
all_cost = np.zeros((n_scenarios,), dtype=np.float32)                     # dispatch cost [$]


@torch.no_grad()
def infer_chunk(chunk_loads):
    """Run inference + bound scaling for one chunk of load scenarios.
    Follows the notebook's test_model (cell 54) without the loss/targets."""
    B = chunk_loads.shape[0]

    # Replicate the constant features for this chunk (notebook cell 36 pattern)
    chunk_bus = np.repeat(samp_grid_bus[np.newaxis, :, :], B, axis=0)
    chunk_gen = np.repeat(samp_grid_generator[np.newaxis, :, :], B, axis=0)
    chunk_shunt = np.repeat(samp_grid_shunt[np.newaxis, :, :], B, axis=0)
    chunk_line = np.repeat(samp_ac_line[np.newaxis, :, :], B, axis=0)
    chunk_tx = np.repeat(samp_transformer[np.newaxis, :, :], B, axis=0)

    # Build HeteroData list for this chunk (notebook create_dataloader body)
    dataset = [
        create_grid_hetero_data(
            chunk_bus, chunk_gen, chunk_loads, chunk_shunt,
            chunk_line, chunk_tx,
            grid_ac_line_senders, grid_ac_line_receivers,
            grid_transformer_senders, grid_transformer_receivers,
            generator_indices, load_indices, shunt_indices,
            bus_pe,
            batch_idx=i,
        )
        for i in range(B)
    ]
    # shuffle=False: results must stay in chronological order
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    v_out, p_out = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        # Forward pass under autocast, as in test_model
        with autocast(device_type=str(device).split(':')[0]):
            pred_dict = model(batch)

            # Rescale sigmoid outputs into their physical bounds and clamp
            voltage_up, voltage_down = convert_voltage_bounds(batch['bus'].x)
            voltages = pred_dict['bus'] * (voltage_up - voltage_down) + voltage_down
            voltages = torch.clamp(voltages, min=voltage_down, max=voltage_up)

            power_up, power_down = convert_power_bounds(batch['generator'].x)
            powers = pred_dict['generator'] * (power_up - power_down) + power_down
            powers = torch.clamp(powers, min=power_down, max=power_up)

        v_out.append(voltages.float().cpu())
        p_out.append(powers.float().cpu())

    # Reshape flat node outputs back to (B, nodes, 2) (notebook cells 66-67)
    v_pred = torch.cat(v_out, dim=0).reshape(B, system_size, 2)
    p_pred = torch.cat(p_out, dim=0).reshape(B, len(generator_indices), 2)
    return v_pred, p_pred


def physics_chunk(chunk_loads, v_pred, p_pred):
    """Violations + cost for one chunk (notebook cells 71-84, 100-112, 85)."""
    B = chunk_loads.shape[0]

    # Scatter loads onto buses (cell 71)
    load_demand = torch.zeros(B, system_size, chunk_loads.shape[-1])
    load_demand[:, load_indices, :] = torch.tensor(chunk_loads, dtype=torch.float32)
    load_demand = load_demand.to(torch.float32)

    # Polar -> rectangular voltages (cell 81)
    complex_v = convert_to_complex_voltage(v_pred)

    # Required net injections + all branch flows (cell 84)
    injection_balance, branch_flows = calculate_only_branch_flows(
        load_demand, complex_v, branch_list,
        Yks, conductance_susceptance, charging_susceptance, Tij_rec,
    )

    # Forward / reverse apparent-power magnitudes (cells 103-107)
    forward_power_flows = torch.cat(
        [branch_flows[key].unsqueeze(dim=1) for key in forward_keys], dim=1)
    reverse_power_flows = torch.cat(
        [branch_flows[key].unsqueeze(dim=1) for key in reverse_keys], dim=1)

    forward_flow_magnitude = convert_to_power_magnitude(forward_power_flows)
    reverse_flow_magnitude = convert_to_power_magnitude(reverse_power_flows)

    # Branch flow violations vs long-term rating (cells 108-109)
    branch_flow_limit = long_term_line_rating.tile((B, 1, 1))
    forward_flow_violations = torch.clamp(forward_flow_magnitude - branch_flow_limit, min=0)
    reverse_flow_violations = torch.clamp(reverse_flow_magnitude - branch_flow_limit, min=0)

    # Scatter predicted generator power onto buses (cell 110);
    # index_add_ correctly sums co-located generators on the same bus
    gen_power = torch.zeros(B, system_size, 2)
    gen_idx_t = torch.tensor(generator_indices)
    for i in range(B):
        gen_power[i, :, :].index_add_(0, gen_idx_t, p_pred[i, :, :])

    # Power balance mismatches (cells 111-112)
    real_mismatch = injection_balance[:, :, 0] - gen_power[:, :, 0]
    reactive_mismatch = injection_balance[:, :, 1] - gen_power[:, :, 1]

    # Dispatch cost in dollars (cost part of compute_optimality, cell 85):
    # sum over generators of c2*Pg^2 + c1*Pg + c0
    p_gens = p_pred[:, :, 0:1]  # select Pg only
    system_metrics = c2 * (p_gens ** 2) + c1 * p_gens + c0
    cost = torch.sum(system_metrics, dim=1).flatten()

    return (real_mismatch, reactive_mismatch,
            forward_flow_violations.squeeze(-1), reverse_flow_violations.squeeze(-1),
            forward_flow_magnitude.squeeze(-1), cost)


# Main loop over the year in chunks
for start in range(0, n_scenarios, batch_size):
    end = min(start + batch_size, n_scenarios)
    chunk_loads = grid_load[start:end]

    v_pred, p_pred = infer_chunk(chunk_loads)
    (real_mm, react_mm, fwd_viol, rev_viol, fwd_flow, cost) = physics_chunk(chunk_loads, v_pred, p_pred)

    # Store chunk results
    all_v_pred[start:end] = v_pred.numpy()
    all_p_pred[start:end] = p_pred.numpy()
    all_pbal_p[start:end] = real_mm.numpy()
    all_pbal_q[start:end] = react_mm.numpy()
    all_fwd_viol[start:end] = fwd_viol.numpy()
    all_rev_viol[start:end] = rev_viol.numpy()
    all_fwd_flow[start:end] = fwd_flow.numpy()
    all_cost[start:end] = cost.numpy()

    print(f'[{end}/{n_scenarios}] done', flush=True)

# ---------------------------------------------------------------------------
# 9. Graph layout for plotting (computed once, deterministic)
# ---------------------------------------------------------------------------
G = nx.Graph()
G.add_nodes_from(range(system_size))
G.add_edges_from([(int(i), int(j)) for (i, j) in branch_list])
# Kamada-Kawai gives a stable, readable layout for meshed grids
pos = nx.kamada_kawai_layout(G)
node_xy = np.array([pos[i] for i in range(system_size)], dtype=np.float32)

# ---------------------------------------------------------------------------
# 10. Save everything the app needs
# ---------------------------------------------------------------------------
np.savez_compressed(
    RESULTS_PATH,
    p_pred=all_p_pred,                    # (T, n_gen, 2) predicted [Pg, Qg], p.u.
    v_pred=all_v_pred,                    # (T, 118, 2) predicted [Va, Vm]
    pbal_p=all_pbal_p,                    # (T, 118) active balance mismatch, p.u.
    pbal_q=all_pbal_q,                    # (T, 118) reactive balance mismatch, p.u.
    fwd_viol=all_fwd_viol,                # (T, n_branch) forward flow violation, p.u.
    rev_viol=all_rev_viol,                # (T, n_branch) reverse flow violation, p.u.
    fwd_flow=all_fwd_flow,                # (T, n_branch) forward |S|, p.u.
    cost=all_cost,                        # (T,) dispatch cost, $
    branch_list=np.array(branch_list, dtype=np.int64),   # (n_branch, 2)
    branch_limit=long_term_line_rating.numpy().flatten(),  # (n_branch,)
    generator_indices=np.asarray(generator_indices, dtype=np.int64),
    load_indices=np.asarray(load_indices, dtype=np.int64),
    node_xy=node_xy,                      # (118, 2) plot coordinates
)
print(f'Saved results to {RESULTS_PATH}')
