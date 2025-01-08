#!/usr/bin/env python3
"""
Integrated Biological RBF PDE Prototype
=======================================

KEY FEATURES:
1. Time-dependent PDE (simple Reaction-Diffusion or "Turing-like" system).
2. Biology-aware node placement with discontinuity detection (tumor boundary).
3. Naive domain decomposition approach using mpi4py (conceptual).
4. RBF-FD for local PDE discretization, refined stencils near boundaries.

DISCLAIMER:
- This code is not tested for large-scale HPC.
- It's an illustrative "big picture" script that merges many concepts.
- More robust data structures, solvers, concurrency, etc. would be required
  for real multi-scale cancer modeling.

HOW TO RUN:
    mpiexec -np 4 python integrated_biological_rbf.py

Or:
    python integrated_biological_rbf.py
(Will just run single-process.)
"""

import numpy as np
import math
from typing import Dict, List, Optional
from dataclasses import dataclass
from mpi4py import MPI  # for naive domain decomposition
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# -------------------------------------------------------------------
# 0) MPI Helpers
# -------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# -------------------------------------------------------------------
# 1) Data Structures and Biological Fields
# -------------------------------------------------------------------
@dataclass
class BioField:
    """Stores a biological scalar field (e.g., cell density, chem gradient)."""
    name: str
    data: np.ndarray  # placeholder array or function-based data

@dataclass
class RBFNode:
    coord: np.ndarray           # (2,) or (3,) for domain dimension
    rbf_params: Dict[str, float] # shape, stencil, etc.
    subdomain_id: int = 0       # which MPI subdomain this node belongs to
    # PDE state variables:
    u: float = 0.0              # primary PDE variable (e.g. concentration)
    v: float = 0.0              # secondary PDE variable if we do a Turing system

@dataclass
class PDEParameters:
    """Define PDE coefficients for reaction-diffusion, or synergy terms."""
    diff_u: float = 0.01   # diffusion coefficient for 'u'
    diff_v: float = 0.005  # diffusion for 'v'
    # Reaction terms for a simple Turing-like system:
    # du/dt = diff_u Lap(u) + alpha*(u - u^3 - v)
    # dv/dt = diff_v Lap(v) + beta*(u - v)
    alpha: float = 1.0
    beta: float = 1.0


# -------------------------------------------------------------------
# 2) BiologicalRegionHandler (Discontinuity + Node Placement)
# -------------------------------------------------------------------
class BiologicalRegionHandler:
    """
    Detects regions of high 'biological activity' or discontinuities
    and places or refines nodes accordingly.
    """
    def __init__(self, discontinuity_threshold=0.2, base_shape=1.0, base_spacing=0.1):
        self.discontinuity_threshold = discontinuity_threshold
        self.base_shape = base_shape
        self.base_spacing = base_spacing

    def detect_biological_transitions(self,
                                      point: np.ndarray,
                                      bio_fields: Dict[str, BioField]) -> Dict[str, float]:
        """
        Estimate local 'gradient' or abrupt changes in each field (toy version).
        Here we just measure relative difference from field's mean as a "discontinuity" proxy.
        """
        gradients = {}
        for field_name, bf in bio_fields.items():
            # Suppose bf.data is a single global array with shape (N,) or so.
            # Real code would do interpolation or a local neighbor check.
            mean_val = np.mean(bf.data) + 1e-12
            local_val = float(np.mean(bf.data))  # placeholder
            rel_diff = abs(local_val - mean_val)/abs(mean_val)
            gradients[field_name] = rel_diff
        return gradients

    def create_transition_layer(self,
                                center_point: np.ndarray,
                                gradients: Dict[str, float]) -> List[RBFNode]:
        """
        Place multiple refined nodes in a small ring around center_point if we detect a high discontinuity.
        """
        transition_nodes = []
        radius = self.base_spacing * 0.5
        n_ring = 4
        max_grad = max(gradients.values())
        shape_scale = 1.0 + max_grad

        for i in range(n_ring):
            angle = 2.0*math.pi*(i/n_ring)
            offset = np.array([radius*math.cos(angle), radius*math.sin(angle)])
            new_pt = center_point + offset
            rbf_params = {
                "shape_parameter": self.base_shape*shape_scale,
                "stencil_type": "one_sided",
                "jump_condition": True
            }
            node = RBFNode(coord=new_pt, rbf_params=rbf_params)
            transition_nodes.append(node)
        return transition_nodes

    def compute_discontinuity_treatment(self, max_grad: float) -> Dict[str, float]:
        # scale shape param by gradient intensity
        shape_param = self.base_shape*(1.0 + max_grad)
        return {
            "shape_parameter": shape_param,
            "stencil_type": "one_sided",
            "jump_condition": True
        }

    def adapt_node_distribution(self,
                                point: np.ndarray,
                                bio_fields: Dict[str, BioField]) -> List[RBFNode]:
        gradients = self.detect_biological_transitions(point, bio_fields)
        max_grad = max(gradients.values())
        if max_grad > self.discontinuity_threshold:
            # create refined layer
            transition_nodes = self.create_transition_layer(point, gradients)
            # main node
            main_params = self.compute_discontinuity_treatment(max_grad)
            main_node = RBFNode(coord=point, rbf_params=main_params)
            transition_nodes.append(main_node)
            return transition_nodes
        else:
            node = RBFNode(coord=point, rbf_params={
                "shape_parameter": self.base_shape,
                "stencil_type": "symmetric",
                "jump_condition": False
            })
            return [node]


# -------------------------------------------------------------------
# 3) Domain Decomposition (Naive)
# -------------------------------------------------------------------
def naive_domain_decomposition(all_coords: np.ndarray) -> np.ndarray:
    """
    Splits the array of points among MPI processes. 
    We'll just slice them in contiguous blocks for demonstration.
    """
    N = len(all_coords)
    chunk_size = N // size
    start = rank * chunk_size
    end = (rank+1)*chunk_size if rank < size-1 else N
    return all_coords[start:end]


# -------------------------------------------------------------------
# 4) Local RBF PDE System
# -------------------------------------------------------------------
class LocalRBFPDESystem:
    """
    Time-dependent PDE (reaction-diffusion Turing-like).
    We'll do RBF-FD for Laplacian, and a simple Euler time-step.
    This is a toy demonstration, not HPC-optimized.
    """

    def __init__(self, nodes: List[RBFNode], pde_params: PDEParameters, dt=0.01, neighbor_radius=0.2):
        self.nodes = nodes
        self.pde_params = pde_params
        self.dt = dt
        self.neighbor_radius = neighbor_radius
        self.kdtree = None
        # We store a "L_u" Laplacian matrix for 'u', and "L_v" for 'v'. 
        self.L_u = None
        self.L_v = None

    def build_spatial_operator(self):
        # build kdtree
        coords = np.array([n.coord for n in self.nodes])
        self.kdtree = cKDTree(coords)
        N = len(self.nodes)

        row_indices_u, col_indices_u, data_vals_u = [], [], []
        row_indices_v, col_indices_v, data_vals_v = [], [], []

        # naive approach for laplacian
        for i, node_i in enumerate(self.nodes):
            # neighbor search
            nbrs = self.kdtree.query_ball_point(node_i.coord, self.neighbor_radius)
            # toy weighting: center=-4, ring=+1
            w = np.zeros(len(nbrs))
            if len(nbrs) > 0:
                w[0] = -4.0
                for jdx in range(1, len(nbrs)):
                    w[jdx] = 1.0

            shape_param = node_i.rbf_params.get("shape_parameter",1.0)
            for loc_j, j in enumerate(nbrs):
                row_indices_u.append(i)
                col_indices_u.append(j)
                # scale by shape_param
                data_vals_u.append(w[loc_j] * shape_param)

                row_indices_v.append(i)
                col_indices_v.append(j)
                # scale similarly
                data_vals_v.append(w[loc_j] * shape_param)

        self.L_u = csr_matrix((data_vals_u, (row_indices_u, col_indices_u)), shape=(N, N))
        self.L_v = csr_matrix((data_vals_v, (row_indices_v, col_indices_v)), shape=(N, N))

    def time_step_euler(self):
        """
        Simple Euler forward step for:
            du/dt = diff_u Lap(u) + alpha*(u - u^3 - v)
            dv/dt = diff_v Lap(v) + beta*(u - v)
        """
        N = len(self.nodes)
        # gather current u, v
        u_array = np.array([nd.u for nd in self.nodes])
        v_array = np.array([nd.v for nd in self.nodes])

        # PDE parameters
        diff_u = self.pde_params.diff_u
        diff_v = self.pde_params.diff_v
        alpha  = self.pde_params.alpha
        beta   = self.pde_params.beta

        # Lap(u), Lap(v)
        Lap_u = self.L_u.dot(u_array)
        Lap_v = self.L_v.dot(v_array)

        # Reaction terms:
        # du/dt = diff_u * Lap(u) + alpha*(u - u^3 - v)
        # dv/dt = diff_v * Lap(v) + beta*(u - v)
        rhs_u = diff_u*Lap_u + alpha*(u_array - u_array**3 - v_array)
        rhs_v = diff_v*Lap_v + beta*(u_array - v_array)

        # Euler step:
        u_new = u_array + self.dt * rhs_u
        v_new = v_array + self.dt * rhs_v

        # update node states
        for i, nd in enumerate(self.nodes):
            nd.u = u_new[i]
            nd.v = v_new[i]


# -------------------------------------------------------------------
# 5) Full Demo
# -------------------------------------------------------------------
def run_biological_rbf_demo():
    # Generate some random domain points globally
    N_global = 40  # total points
    if rank==0: print(f"[INFO] Generating {N_global} random points in a 2D domain...")
    global_coords = np.random.rand(N_global, 2)*1.0  # in [0,1]^2

    # naive domain decomposition
    local_coords = naive_domain_decomposition(global_coords)

    # Example biological fields
    # We'll create random data arrays for "chem_gradient" & "cell_density"
    # In real usage, you'd load actual experimental data or PDE solutions.
    # Just keep them local for the sake of demonstration, or gather them first.
    # For simplicity, let's create them globally then scatter if needed.
    # We'll pretend we have them on every rank for now.

    chem_grad_global = np.random.rand(N_global)
    cell_density_global = np.random.rand(N_global)

    # If needed, we can do a naive slice for local
    start_idx = rank*(N_global//size)
    end_idx   = (rank+1)*(N_global//size) if rank < size-1 else N_global

    local_chem = chem_grad_global[start_idx:end_idx]
    local_cell = cell_density_global[start_idx:end_idx]

    # Create dictionary
    bio_fields = {
        "chem_grad": BioField("chem_grad", local_chem),
        "cell_density": BioField("cell_density", local_cell)
    }

    # Build node distribution
    region_handler = BiologicalRegionHandler(
        discontinuity_threshold=0.3,
        base_shape=1.0,
        base_spacing=0.1
    )

    local_nodes = []
    for pt in local_coords:
        new_ns = region_handler.adapt_node_distribution(pt, bio_fields)
        for nd in new_ns:
            nd.subdomain_id = rank
            # Initialize PDE states (u, v)
            nd.u = np.random.rand()*0.1
            nd.v = np.random.rand()*0.1
        local_nodes.extend(new_ns)

    # Build PDE system
    pde_params = PDEParameters(
        diff_u=0.01,
        diff_v=0.005,
        alpha=1.0,
        beta=1.0
    )
    rbf_system = LocalRBFPDESystem(nodes=local_nodes, pde_params=pde_params, dt=0.01, neighbor_radius=0.15)
    rbf_system.build_spatial_operator()

    # We do a few time steps
    nsteps = 20
    if rank==0: print(f"[INFO] Starting time integration for {nsteps} steps (Euler) ...")
    for step in range(nsteps):
        rbf_system.time_step_euler()
        if step%5==0 and rank==0:
            print(f"  Step {step} done...")

    # Each process now has a local solution array. In principle,
    # you could gather them to rank 0 for visualization or further analysis.
    # We'll do a naive gather:
    local_uv = [(nd.coord, nd.u, nd.v) for nd in local_nodes]
    all_data = comm.gather(local_uv, root=0)

    if rank==0:
        # Flatten
        all_flat = []
        for chunk in all_data:
            all_flat.extend(chunk)
        # Sort them by x,y or something if we want
        # Just print first few
        print("[RESULT] Combined PDE results (first 10):")
        for i, item in enumerate(all_flat[:10]):
            coord, u, v = item
            print(f"  {i}: coord={coord}, u={u:.4f}, v={v:.4f}")


if __name__ == "__main__":
    run_biological_rbf_demo()
    comm.Barrier()
    if rank==0:
        print("[INFO] Done with integrated RBF PDE demonstration.")
