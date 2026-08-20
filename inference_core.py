"""
inference_core.py
=================
Model classes and physics functions for HH-MPNN ACOPF inference on the 118-bus
system. All classes/functions are lifted directly from the training notebook
(ML4ACOPF_HybridHeteroGNN_PGLearn.ipynb) with only the minimal changes needed
for standalone inference. Every deviation from the notebook is marked with a
"# CHANGED:" comment.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
# CHANGED: use PyG's native scatter instead of the separate torch_scatter
# package; scatter(..., reduce='sum') is numerically identical to
# torch_scatter.scatter_add
from torch_geometric.utils import to_dense_batch, scatter
from torch_geometric.nn.attention import PerformerAttention


# ---------------------------------------------------------------------------
# Positional-encoding pipeline (notebook cells 16-28)
# ---------------------------------------------------------------------------

def compute_gandb(edge_inputs):
    """Compute series conductance g and susceptance b from branch r and x.

    CHANGED: the notebook version referenced the global `edge_inputs` inside
    the function body (its parameter was misspelled `edge_iputs`); here the
    parameter is actually used. The computation itself is identical.
    """
    line_r = edge_inputs[:, 4:5]   # series resistance r
    line_x = edge_inputs[:, 5:6]   # series reactance x

    line_g = line_r / (line_r**2 + line_x**2)
    line_b = -line_x / (line_r**2 + line_x**2)

    return line_g, line_b


def get_B_matrix(N, edges, edge_weights):
    """Build a symmetric weighted adjacency matrix from branch susceptances."""
    # Create a zero tensor of shape (N,N)
    B_matrix = torch.zeros((N, N)).to(torch.float64)

    # Unpack the edges into source and destination nodes
    sources, destinations = zip(*edges)

    # Use advanced indexing to place weights in the right spots
    B_matrix[sources, destinations] = edge_weights.squeeze()
    B_matrix[destinations, sources] = edge_weights.squeeze()
    return B_matrix


def adjacency_to_laplacian(B_adj):
    """Convert a weighted adjacency matrix into a graph Laplacian."""
    # Ensure matrix is square
    assert B_adj.shape[0] == B_adj.shape[1], "Input must be square"

    # Copy to avoid modifying original
    B_laplacian = B_adj.copy()

    # Set diagonal as row sum of adjacency (i.e., degree)
    np.fill_diagonal(B_laplacian, -B_adj.sum(axis=1))

    return B_laplacian


def effective_resistance_matrix(b_mat):
    """
    Computes the effective resistance matrix from a susceptance Laplacian.

    Parameters:
    b_mat (ndarray): weighted laplacian matrix

    Returns:
    ndarray: Effective resistance matrix (N x N)
    """
    L = b_mat
    n = L.shape[0]

    # Remove reference node (last row and column) to deal with singularity
    keep = np.arange(n - 1)
    L_reduced = L[np.ix_(keep, keep)]

    # Invert reduced Laplacian
    L_reduced_inv = np.linalg.inv(L_reduced)

    # Expand to full pseudoinverse
    L_plus = np.zeros((n, n))
    L_plus[np.ix_(keep, keep)] = L_reduced_inv

    # Project to orthogonal component (to make it true pseudoinverse)
    I = np.eye(n)
    ones = np.ones((n, n)) / n
    L_plus = (I - ones) @ L_plus @ (I - ones)

    # Compute resistance: R_ij = L^+_ii + L^+_jj - 2L^+_ij
    diag = np.diag(L_plus)
    R = diag[:, None] + diag[None, :] - 2 * L_plus
    return R


def compute_row_statistics_vectorized(resistance_matrix):
    """
    Row statistics of the effective-resistance matrix, excluding the diagonal.
    Returns array of shape (N, 5): [mean, median, std, max, min] per bus.
    These 5 statistics are the positional encodings appended to node embeddings.
    """
    N = resistance_matrix.shape[0]

    # Create a mask to exclude diagonal elements
    mask = ~np.eye(N, dtype=bool)

    # Initialize result matrix
    stats_matrix = np.zeros((N, 5))

    # For each row, extract non-diagonal elements and compute statistics
    for i in range(N):
        row_no_diag = resistance_matrix[i, mask[i]]

        stats_matrix[i, 0] = np.mean(row_no_diag)
        stats_matrix[i, 1] = np.median(row_no_diag)
        stats_matrix[i, 2] = np.std(row_no_diag)
        stats_matrix[i, 3] = np.max(row_no_diag)
        stats_matrix[i, 4] = np.min(row_no_diag)

    return stats_matrix


# ---------------------------------------------------------------------------
# Model definition (notebook cells 45-48) — unchanged
# ---------------------------------------------------------------------------

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True, use_leaky=False):
        super().__init__()
        # Use Sequential instead of ModuleList for faster forward pass
        modules = []
        for i in range(layers):
            modules.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                modules.append(torch.nn.ReLU())
            if use_leaky:
                modules.append(torch.nn.LeakyReLU(negative_slope=0.02))
        if layernorm:
            modules.append(torch.nn.LayerNorm(output_size))

        self.network = torch.nn.Sequential(*modules)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.network:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        # Sequential is faster than iterating through ModuleList
        return self.network(x)


class HeteroPerformerLayer(nn.Module):
    """Global attention layer applied across all node types in a batch."""

    def __init__(self, hidden_dim, num_heads=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attn = PerformerAttention(
            channels=hidden_dim,
            heads=num_heads
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Post-attention MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x_dict, xm_dict, batch_dict):
        # Flatten all node types
        flat_x, flat_xm, flat_batch = [], [], []
        slices = {}
        offset = 0

        for ntype in x_dict:
            x = x_dict[ntype]
            xm = xm_dict[ntype]
            b = batch_dict[ntype]  # batch indices for each node

            slices[ntype] = slice(offset, offset + x.size(0))
            flat_x.append(x)
            flat_xm.append(xm)
            flat_batch.append(b)
            offset += x.size(0)

        x_all = torch.cat(flat_x, dim=0)                # [N, D]
        xm_all = torch.cat(flat_xm, dim=0)              # [N, D]
        global_batch_all = torch.cat(flat_batch, dim=0)  # [N]

        _, batch_all = torch.unique(global_batch_all, return_inverse=True)
        # resort batch index to be ascending, so the xs must be sorted the same way
        sorted_indices = torch.argsort(batch_all)
        batch_all_sorted = batch_all[sorted_indices]
        x_all_sorted = x_all[sorted_indices]
        xm_all_sorted = xm_all[sorted_indices]

        # Convert to [B, N_max, D] and mask
        x_dense, mask = to_dense_batch(x_all_sorted, batch_all_sorted)  # [B, N, D], [B, N]

        # Apply masked Performer attention
        x_attn = self.attn(x_dense, mask=mask)  # [B, N, D]
        # Residual + Norm
        xt_out = self.norm1(x_dense + x_attn)
        # Unpad: [real_nodes, D]
        xt_out = xt_out[mask]

        x_comb = self.norm2(xt_out + xm_all_sorted)
        x_final = self.mlp(x_comb)

        # restore the original (unsorted) node arrangement
        unsorted_x = torch.empty_like(x_final)
        unsorted_x[sorted_indices] = x_final
        x_final = unsorted_x

        # Unflatten by slice
        return {ntype: x_final[slices[ntype]] for ntype in x_dict.keys()}


class HeteroInteractionNetwork(nn.Module):
    """One message-passing layer over the heterogeneous grid graph."""

    def __init__(self, node_types, edge_types, physical_edge_types, hidden_size, layers):
        super().__init__()

        self.physical_edge_types = physical_edge_types
        self.edge_updaters = nn.ModuleDict()
        for src, rel, dst in edge_types:
            edge_key = f"{src}_{rel}_{dst}"
            edge_type = (src, rel, dst)

            # For physical edges (ac_line and transformer), use node + edge features
            if edge_type in physical_edge_types:
                self.edge_updaters[edge_key] = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
            else:
                # For other edge types, only use node features
                self.edge_updaters[edge_key] = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

        # Create a node updater for each node type
        self.node_updaters = nn.ModuleDict({
            node_type: MLP(hidden_size * 2, hidden_size, hidden_size, layers)
            for node_type in node_types
        })

    def forward(self, x_dict, edge_indices_dict, edge_features_dict):
        # Store updated node and edge features
        updated_edge_features = {}

        # Prepare aggregated messages storage
        aggregated_messages = {node_type: torch.zeros_like(feat)
                               for node_type, feat in x_dict.items()}

        # Process each edge type in parallel
        for edge_type, edge_index in edge_indices_dict.items():
            src_type, rel_type, dst_type = edge_type
            edge_key = f"{src_type}_{rel_type}_{dst_type}"

            # Get node features for this edge
            src, dst = edge_index
            x_i = x_dict[dst_type][dst]  # Destination nodes
            x_j = x_dict[src_type][src]  # Source nodes

            # Update edge features based on edge type
            if edge_type in self.physical_edge_types:
                # For physical edges, include edge features in message
                edge_feature = edge_features_dict[edge_type]
                edge_msg = torch.cat((x_i, x_j, edge_feature), dim=-1)
                updated_edge = self.edge_updaters[edge_key](edge_msg)
                # residual connection on physical edge features
                updated_edge = updated_edge + edge_feature
                updated_edge_features[edge_type] = updated_edge
            else:
                # For non-physical edges, only use node features
                edge_msg = torch.cat((x_i, x_j), dim=-1)
                updated_edge = self.edge_updaters[edge_key](edge_msg)
                # no residual for non-physical edges (no initial edge features)
                updated_edge_features[edge_type] = updated_edge

            # dtype alignment (needed under mixed precision)
            aggregated_messages[dst_type] = aggregated_messages[dst_type].to(updated_edge.dtype)
            aggregated_messages[src_type] = aggregated_messages[src_type].to(updated_edge.dtype)
            # CHANGED: message aggregation via PyG's native scatter.
            # torch_scatter.scatter_add accumulated in-place through its
            # `out=` argument; PyG's scatter has no `out=`, so the running
            # per-node-type totals are accumulated with an explicit addition
            # instead. dim_size pins the output to the full node count of the
            # type, so nodes receiving no messages from this edge type keep
            # their zero rows (same behaviour as before).
            aggregated_messages[dst_type] = aggregated_messages[dst_type] + scatter(
                updated_edge, dst, dim=0,
                dim_size=x_dict[dst_type].shape[0], reduce='sum')
            aggregated_messages[src_type] = aggregated_messages[src_type] + scatter(
                updated_edge, src, dim=0,
                dim_size=x_dict[src_type].shape[0], reduce='sum')

        # Update node features
        updated_nodes = {}
        for node_type, x in x_dict.items():
            # Combine node features with aggregated messages
            node_input = torch.cat((x, aggregated_messages[node_type]), dim=-1)
            node_update = self.node_updaters[node_type](node_input)
            updated_nodes[node_type] = x + node_update  # Residual connection

        return updated_nodes, updated_edge_features


class HeteroInteractGNN(torch.nn.Module):
    """Hybrid Heterogeneous MPNN: interleaved local message passing and
    global Performer attention, with sigmoid-bounded PQVT outputs."""

    def __init__(
        self,
        hidden_size=256,
        n_mp_layers=5,
        bus_features=4,
        gen_features=11,
        load_features=2,
        shunt_features=2,
        ac_line_features=9,
        transformer_features=11,
        connects_to_features=3,
        output_dim=2
    ):
        super().__init__()

        # Define node and edge types
        self.node_types = ['bus', 'generator', 'load', 'shunt']
        self.edge_types = [
            ('bus', 'ac_line', 'bus'),
            ('bus', 'transformer', 'bus'),
            ('generator', 'connects_to', 'bus'),
            ('load', 'connects_to', 'bus'),
            ('shunt', 'connects_to', 'bus')
        ]

        self.physical_edge_types = [
            ('bus', 'ac_line', 'bus'),
            ('bus', 'transformer', 'bus')
        ]

        # Node encoders - separate MLP for each node type.
        # Output is hidden_size-5 so that the 5 PE statistics can be concatenated.
        self.node_encoders = nn.ModuleDict({
            'bus': MLP(bus_features, hidden_size, hidden_size - 5, 2),
            'generator': MLP(gen_features, hidden_size, hidden_size - 5, 2),
            'load': MLP(load_features, hidden_size, hidden_size - 5, 2),
            'shunt': MLP(shunt_features, hidden_size, hidden_size - 5, 2)
        })

        self.global_attn_layers = nn.ModuleList([
            HeteroPerformerLayer(hidden_size)
            for _ in range(n_mp_layers)
        ])

        # Edge encoders - separate MLP for each physical edge type
        self.edge_encoders = nn.ModuleDict({
            'ac_line': MLP(ac_line_features, hidden_size, hidden_size, 2),
            'transformer': MLP(transformer_features, hidden_size, hidden_size, 2)
        })

        # Interaction network layers
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([
            HeteroInteractionNetwork(self.node_types, self.edge_types, self.physical_edge_types, hidden_size, 2)
            for _ in range(n_mp_layers)
        ])

        # Node decoders - separate for bus and generator
        self.node_decoders = nn.ModuleDict({
            'bus': MLP(hidden_size, hidden_size, output_dim, 2, layernorm=False),
            'generator': MLP(hidden_size, hidden_size, output_dim, 2, layernorm=False)
        })

    def forward(self, data):
        # Encode node features
        x_dict_init = {}
        x_dict = {}

        # Batch node encoding, then append the 5-dim positional encodings
        for node_type in self.node_types:
            if hasattr(data[node_type], 'x'):
                x_dict_init[node_type] = self.node_encoders[node_type](data[node_type].x)
                x_dict[node_type] = torch.cat([x_dict_init[node_type], data[node_type].pe], dim=-1)

        # Encode edge features
        edge_feature_dict = {}
        for src, rel, dst in self.edge_types:
            edge_type = (src, rel, dst)
            if (edge_type in self.physical_edge_types and
                    edge_type in data.edge_types and
                    hasattr(data[edge_type], 'edge_attr')):
                edge_feature_dict[edge_type] = self.edge_encoders[rel](data[edge_type].edge_attr)

        # Extract edge indices
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in self.edge_types
            if edge_type in data.edge_types and hasattr(data[edge_type], 'edge_index')
        }

        batch_dict = {
            ntype: data[ntype].batch
            for ntype in x_dict
        }

        # Apply message passing layers, interleaved with global attention
        for i in range(self.n_mp_layers):
            xm_dict, edge_feature_dict = self.layers[i](x_dict, edge_index_dict, edge_feature_dict)
            x_dict = self.global_attn_layers[i](x_dict, xm_dict, batch_dict)

        # Apply decoders (sigmoid gives outputs in [0,1], later rescaled to bounds)
        output = {}
        output['bus'] = torch.sigmoid(self.node_decoders['bus'](x_dict['bus']))
        output['generator'] = torch.sigmoid(self.node_decoders['generator'](x_dict['generator']))

        return output


# ---------------------------------------------------------------------------
# Bound conversion (notebook cells 50-51)
# ---------------------------------------------------------------------------
# CHANGED: the notebook versions used the global `device`; here the device is
# taken from the input tensor so the functions work on CPU or GPU unchanged.

def convert_voltage_bounds(model_input):
    """Extract per-bus voltage bounds; theta bounded to [-2, 2] rad."""
    num_nodes = model_input.shape[0]
    dev = model_input.device  # CHANGED: infer device from input

    vmin = model_input[:, 2:3]
    vmax = model_input[:, 3:4]

    thetamin = torch.tensor([-2.00]).to(dev)
    thetamin = thetamin.tile((num_nodes, 1))

    thetamax = torch.tensor([2.00]).to(dev)
    thetamax = thetamax.tile((num_nodes, 1))

    bounds_up = torch.concat((thetamax, vmax), dim=1)
    bounds_down = torch.concat((thetamin, vmin), dim=1)

    return bounds_up, bounds_down


def convert_power_bounds(model_input):
    """Extract per-generator active/reactive power bounds from features."""
    pmin = model_input[:, 2:3]
    pmax = model_input[:, 3:4]

    qmin = model_input[:, 5:6]
    qmax = model_input[:, 6:7]

    bounds_up = torch.concat((pmax, qmax), dim=1)
    bounds_down = torch.concat((pmin, qmin), dim=1)

    return bounds_up, bounds_down


# ---------------------------------------------------------------------------
# HeteroData construction (notebook cell 31)
# ---------------------------------------------------------------------------
# CHANGED: `bus_pe` is now an explicit argument instead of a notebook global,
# and solution targets are optional (not needed at inference time).

def create_grid_hetero_data(
    grid_bus,                    # (B, 118, 4)
    grid_generator,              # (B, n_gen, 11)
    grid_load,                   # (B, n_load, 2)
    grid_shunt,                  # (B, n_shunt, 2)
    grid_ac_line_features,       # (B, n_lines, 9)
    grid_transformer_features,   # (B, n_tx, 11)
    grid_ac_line_senders,        # (n_lines, 1)
    grid_ac_line_receivers,      # (n_lines, 1)
    grid_transformer_senders,    # (n_tx, 1)
    grid_transformer_receivers,  # (n_tx, 1)
    generator_indices,           # Indices connecting generators to buses
    load_indices,                # Indices connecting loads to buses
    shunt_indices,               # Indices connecting shunts to buses
    bus_pe,                      # (118, 5) positional encodings  # CHANGED: passed in
    batch_idx=0                  # Batch index to extract
):
    # Create a new HeteroData instance
    data = HeteroData()

    # Extract features for the specified batch
    bus_features = torch.tensor(grid_bus[batch_idx], dtype=torch.float)
    generator_features = torch.tensor(grid_generator[batch_idx], dtype=torch.float)
    load_features = torch.tensor(grid_load[batch_idx], dtype=torch.float)
    shunt_features = torch.tensor(grid_shunt[batch_idx], dtype=torch.float)

    ac_line_features = torch.tensor(grid_ac_line_features[batch_idx], dtype=torch.float)
    transformer_features = torch.tensor(grid_transformer_features[batch_idx], dtype=torch.float)

    # Add node features
    data['bus'].x = bus_features
    data['generator'].x = generator_features
    data['load'].x = load_features
    data['shunt'].x = shunt_features

    # Add node positional encoding (effective-resistance row statistics)
    data['bus'].pe = bus_pe
    data['generator'].pe = bus_pe[generator_indices]
    data['load'].pe = bus_pe[load_indices]
    data['shunt'].pe = bus_pe[shunt_indices]

    # CHANGED: no .y targets at inference time (the notebook attached
    # solution_bus / solution_generator here for supervised training).

    # Add edge indices and features for AC lines (bus to bus)
    senders = torch.tensor(grid_ac_line_senders.flatten(), dtype=torch.long)
    receivers = torch.tensor(grid_ac_line_receivers.flatten(), dtype=torch.long)
    edge_index = torch.stack([senders, receivers], dim=0)
    data['bus', 'ac_line', 'bus'].edge_index = edge_index
    data['bus', 'ac_line', 'bus'].edge_attr = ac_line_features

    # Add edge indices and features for transformers (bus to bus)
    senders = torch.tensor(grid_transformer_senders.flatten(), dtype=torch.long)
    receivers = torch.tensor(grid_transformer_receivers.flatten(), dtype=torch.long)
    edge_index = torch.stack([senders, receivers], dim=0)
    data['bus', 'transformer', 'bus'].edge_index = edge_index
    data['bus', 'transformer', 'bus'].edge_attr = transformer_features

    # Add pseudo-edges from generators to buses
    gen_to_bus = torch.tensor(generator_indices.flatten(), dtype=torch.long)
    gen_indices = torch.arange(len(gen_to_bus), dtype=torch.long)
    gen_edge_index = torch.stack([gen_indices, gen_to_bus], dim=0)
    data['generator', 'connects_to', 'bus'].edge_index = gen_edge_index
    data['generator', 'connects_to', 'bus'].edge_attr = torch.ones((len(gen_to_bus), 3))

    # Add pseudo-edges from loads to buses
    load_to_bus = torch.tensor(load_indices.flatten(), dtype=torch.long)
    load_arange = torch.arange(len(load_to_bus), dtype=torch.long)  # CHANGED: renamed to avoid shadowing load_indices
    load_edge_index = torch.stack([load_arange, load_to_bus], dim=0)
    data['load', 'connects_to', 'bus'].edge_index = load_edge_index
    data['load', 'connects_to', 'bus'].edge_attr = torch.ones((len(load_to_bus), 3))

    # Add pseudo-edges from shunts to buses
    shunt_to_bus = torch.tensor(shunt_indices.flatten(), dtype=torch.long)
    shunt_arange = torch.arange(len(shunt_to_bus), dtype=torch.long)  # CHANGED: renamed to avoid shadowing shunt_indices
    shunt_edge_index = torch.stack([shunt_arange, shunt_to_bus], dim=0)
    data['shunt', 'connects_to', 'bus'].edge_index = shunt_edge_index
    data['shunt', 'connects_to', 'bus'].edge_attr = torch.ones((len(shunt_to_bus), 3))

    return data


# ---------------------------------------------------------------------------
# Physics: branch flows and injections (notebook cells 77-79, 106)
# ---------------------------------------------------------------------------

def calculate_only_branch_flows(
    demand: torch.Tensor,   # shape (batch_size, N, 2) [real, imag]
    voltage: torch.Tensor,  # shape (batch_size, N, 2) [real, imag]
    branches: list,         # list of tuples (from_bus, to_bus)
    Yks: torch.Tensor,      # shape (batch_size, N, 2) [real, imag] shunt admittance
    Yij: torch.Tensor,      # shape (n_branch, 2) [real, imag] branch admittance
    Yijc: torch.Tensor,     # shape (n_branch, 2) [real, imag] branch charging admittance
    Tij: torch.Tensor,      # shape (n_branch, 2) [real, imag] transformation ratio
) -> torch.Tensor:
    batch_size = demand.shape[0]
    num_nodes = voltage.shape[1]

    # Helper function for batched complex multiplication
    def complex_mult_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
        ], dim=-1)

    # Helper function for batched complex conjugate
    def complex_conj_batch(x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x[..., 0], -x[..., 1]], dim=-1)

    # Helper function for batched complex division
    def complex_div_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        denominator = b[..., 0]**2 + b[..., 1]**2
        return torch.stack([
            (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]) / denominator,
            (a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1]) / denominator
        ], dim=-1)

    # Initialize generator power tensor
    generator_power = torch.zeros_like(demand)

    # Calculate shunt power terms for each node (vectorized)
    v_mag_sq = torch.sum(voltage**2, dim=-1, keepdim=True)               # (B, N, 1)
    v_mag_sq = torch.cat([v_mag_sq, torch.zeros_like(v_mag_sq)], dim=-1)  # (B, N, 2)

    # Broadcast Yks to match batch dimension
    Yks_batch = Yks.expand(batch_size, -1, -1)  # (B, N, 2)

    shunt_power = complex_mult_batch(complex_conj_batch(Yks_batch), v_mag_sq)

    # Pre-allocate branch flows dictionary with tensors
    branch_flows = {}

    # Calculate branch flows (vectorized over the batch dimension)
    for idx, (i, j) in enumerate(branches):

        # Get complex voltage at both ends
        vi = voltage[:, i]  # (B, 2)
        vj = voltage[:, j]  # (B, 2)

        # First term calculations
        vi_mag_sq = torch.sum(vi**2, dim=-1, keepdim=True)                     # (B, 1)
        vi_mag_sq = torch.cat([vi_mag_sq, torch.zeros_like(vi_mag_sq)], dim=-1)  # (B, 2)

        tij_mag_sq = torch.sum(Tij[idx]**2).unsqueeze(0)
        tij_mag_sq_tensor = torch.tensor([tij_mag_sq, 0.0], dtype=torch.float32).expand(batch_size, -1)

        vi_over_tij_sq = complex_div_batch(vi_mag_sq, tij_mag_sq_tensor)

        # Sum of branch admittance and charging admittance
        Y_total = torch.stack([
            Yij[idx, 0] + Yijc[idx, 0],
            Yij[idx, 1] + Yijc[idx, 1]
        ]).expand(batch_size, -1)

        term1 = complex_mult_batch(complex_conj_batch(Y_total), vi_over_tij_sq)

        # Second term calculations
        vivj = complex_mult_batch(vi, complex_conj_batch(vj))
        term2 = complex_mult_batch(
            complex_conj_batch(Yij[idx].expand(batch_size, -1)),
            complex_div_batch(vivj, Tij[idx].expand(batch_size, -1))
        )

        # Total branch flow Sij (sending-end apparent power, from i to j)
        Sij = term1 - term2
        branch_flows[(i, j, idx)] = Sij

        # Reverse flow calculations (from j to i)
        vj_mag_sq = torch.sum(vj**2, dim=-1, keepdim=True)
        vj_mag_sq = torch.cat([vj_mag_sq, torch.zeros_like(vj_mag_sq)], dim=-1)

        term1_ji = complex_mult_batch(complex_conj_batch(Y_total), vj_mag_sq)
        vjvi = complex_mult_batch(complex_conj_batch(vi), vj)
        term2_ji = complex_mult_batch(
            complex_conj_batch(Yij[idx].expand(batch_size, -1)),
            complex_div_batch(vjvi, complex_conj_batch(Tij[idx].expand(batch_size, -1)))
        )

        Sji = term1_ji - term2_ji
        branch_flows[(j, i, idx)] = Sji

    # Aggregate net required injection for each node:
    # sum of outgoing branch flows + demand + shunt consumption
    for i in range(num_nodes):
        for index, (from_bus, to_bus) in enumerate(branches):
            if from_bus == i:
                generator_power[:, i] += branch_flows[(from_bus, to_bus, index)]
            if to_bus == i:
                generator_power[:, i] += branch_flows[(to_bus, from_bus, index)]

    generator_power += demand + shunt_power

    return generator_power, branch_flows


def convert_to_complex_voltage(voltage_tensor):
    """(B, N, 2) [Va, Vm] polar -> (B, N, 2) [real, imag] rectangular."""
    # Extract angle and magnitude
    voltage_angle = voltage_tensor[:, :, 0:1]  # In radians
    voltage_magnitude = voltage_tensor[:, :, 1:]

    # Calculate real and imaginary parts
    real_voltage = voltage_magnitude * torch.cos(voltage_angle)
    imaginary_voltage = voltage_magnitude * torch.sin(voltage_angle)

    return torch.concat((real_voltage, imaginary_voltage), dim=2)


def convert_to_complex_rectangle(tensor_2d):
    """(n, 2) [magnitude, angle] polar -> (n, 2) [real, imag] rectangular."""
    # Extract magnitude and angle
    tensor_mag = tensor_2d[:, 0:1]
    tensor_angle = tensor_2d[:, 1:]

    # Calculate real and imaginary parts
    real_tensor = tensor_mag * torch.cos(tensor_angle)
    imaginary_tensor = tensor_mag * torch.sin(tensor_angle)

    return torch.concat((real_tensor, imaginary_tensor), dim=1)


def convert_to_power_magnitude(power_flow):
    """|S| = sqrt(P^2 + Q^2) with a trailing singleton dim, as in the notebook."""
    real = power_flow[..., 0]
    imag = power_flow[..., 1]

    # Compute the magnitudes
    magnitudes = torch.sqrt(real**2 + imag**2)

    result_tensor = magnitudes.unsqueeze(-1)

    return result_tensor
