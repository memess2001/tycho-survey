"""The instrument: refined information (RI) on hierarchical log-linear models.

Modules: :mod:`~tycho.ri.hypergraph` (families of subsets), :mod:`~tycho.ri.loglinear` (tables, m-projection,
information measures, binary coordinates), :mod:`~tycho.ri.refined` (chain and local RI),
:mod:`~tycho.ri.estimators` (finite-sample estimation, scans, FDR), :mod:`~tycho.ri.discretize` (binarisation),
:mod:`~tycho.ri.variables` (:class:`VariableSet`, the contract type). All information values are in nats.
"""
from tycho.ri.discretize import binarize, select_informative
from tycho.ri.estimators import (
    RIEstimate,
    ScanResult,
    bh_fdr,
    candidate_subsets,
    estimate_local_ri,
    ri_df,
    ri_scan,
)
from tycho.ri.hypergraph import Subset
from tycho.ri.loglinear import (
    ProjectionResult,
    entropy,
    eta_binary,
    joint_table,
    kl,
    loglinear_table,
    m_project,
    marginal,
    mutual_information,
    theta_01_to_spin,
    theta_binary,
    theta_spin_to_01,
    total_correlation,
)
from tycho.ri.refined import (
    ChainDecomposition,
    RIStep,
    chain_decomposition,
    graded_chain,
    greedy_selection,
    local_projection_batch,
    local_ri,
    local_ri_batch,
    local_ri_from_table,
)
from tycho.ri.variables import VariableSet

__all__ = [
    # SPEC §2.7
    "VariableSet",
    "m_project",
    "kl",
    "local_ri",
    "local_ri_batch",
    "chain_decomposition",
    "graded_chain",
    "greedy_selection",
    "estimate_local_ri",
    "ri_scan",
    "bh_fdr",
    "binarize",
    # extras
    "Subset",
    "ProjectionResult",
    "joint_table",
    "marginal",
    "entropy",
    "mutual_information",
    "total_correlation",
    "loglinear_table",
    "theta_binary",
    "eta_binary",
    "theta_spin_to_01",
    "theta_01_to_spin",
    "RIStep",
    "ChainDecomposition",
    "local_projection_batch",
    "local_ri_from_table",
    "RIEstimate",
    "ScanResult",
    "ri_df",
    "candidate_subsets",
    "select_informative",
]
