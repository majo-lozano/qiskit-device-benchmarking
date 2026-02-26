from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union
import copy
import numpy as np
import matplotlib.pyplot as plt
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from pandas import DataFrame

from qiskit import QuantumCircuit
from qiskit.result import marginal_counts as mcts

from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.models import BackendProperties

from qiskit_experiments.framework import BatchExperiment, ParallelExperiment
from qiskit_experiments.library import StandardRB
from qiskit_experiments.library.randomized_benchmarking import LayerFidelity
from qiskit_experiments.library import T1, T2Hahn

import qiskit_device_benchmarking.utilities.graph_utils as gu
import qiskit_device_benchmarking.utilities.layer_fidelity_utils as lfu


# ---------------------- Helper functions ---------------------- #
def _build_readout_circuits(num_qubits: int) -> List[QuantumCircuit]:
    """Create a SPAM experiment on all qubits on parallel."""
    spam0 = QuantumCircuit(num_qubits, num_qubits)
    spam0.measure_all()
    spam1 = QuantumCircuit(num_qubits, num_qubits)
    spam1.x(range(num_qubits))
    spam1.measure_all()
    return [spam0, spam1]


def _get_twoq_gate(backend: IBMBackend) -> str:
    """Get native 2Q gate of a device"""
    basis = backend.configuration().basis_gates
    if "ecr" in basis:
        return "ecr"
    if "cz" in basis:
        return "cz"
    return "cx"


def _get_oneq_basis(backend: IBMBackend) -> List[str]:
    """Get one-qubit basis gates excluding the chosen two-qubit gate; skip rx/rzz."""
    oneq = []
    twoq_gate = _get_twoq_gate(backend)
    for g in backend.configuration().basis_gates:
        if g.casefold() in {"rx", "rzz", "xslow"}:
            continue
        if g.casefold() != twoq_gate.casefold():
            oneq.append(g)
    return oneq


def _build_oneq_rb_experiments(
    backend: IBMBackend,
    lengths: np.ndarray,
    num_samples: int,
    seed: int,
    samples_m: int,
) -> BatchExperiment:
    """Standard 1Q RB across independent qubit sets."""
    G = backend.coupling_map.graph.to_undirected(multigraph=False)
    sqrb_batches = gu.get_iso_qubit_list(G)

    sqrb_exp_list: List[ParallelExperiment] = []
    for batch in sqrb_batches:
        rb1q_exps = []
        for q in batch:
            rb1q_exps.append(
                StandardRB(
                    physical_qubits=[int(q)],
                    lengths=lengths,
                    backend=backend,
                    seed=seed,
                    num_samples=num_samples,
                )
            )
        sqrb_exp_list.append(ParallelExperiment(rb1q_exps, backend=backend, flatten_results=True))

    sqrb_exp = BatchExperiment(sqrb_exp_list, backend=backend, flatten_results=True)
    sqrb_exp.set_experiment_options(separate_jobs=True)
    sqrb_exp.experiment_options.max_circuits = samples_m * num_samples
    return sqrb_exp


def _build_layered_rb_experiments(
    backend: IBMBackend,
    lengths: List[int],
    num_samples: int,
    seed: int,
    max_circuits: int,
) -> Tuple[LayerFidelity, LayerFidelity, List[List[Tuple[int, int]]]]:
    """Layered RB (LayerFidelity) for horizontal & vertical grids; returns (lf_h, lf_v, layers)."""
    twoq_gate = _get_twoq_gate(backend)
    oneq_gates = _get_oneq_basis(backend)

    grid_chains = lfu.get_grids(backend)
    coupling_map = backend.coupling_map
    edges = list(backend.target[twoq_gate].keys())

    layers: List[List[Tuple[int, int]]] = [[] for _ in range(4)]
    grid_chain_flt = [[], []]

    for i in range(2):  # 0=H, 1=V
        all_pairs = gu.path_to_edges(grid_chains[i], coupling_map)
        for j, pair_lst in enumerate(all_pairs):
            grid_chain_flt[i] += grid_chains[i][j]
            sub_pairs = [tuple(p) if tuple(p) in edges else tuple(p)[::-1] for p in pair_lst]
            layers[2 * i] += sub_pairs[0::2]
            layers[2 * i + 1] += sub_pairs[1::2]

    h_qubits = grid_chain_flt[0]
    v_qubits = grid_chain_flt[1]

    lf_h = LayerFidelity(
        physical_qubits=h_qubits,
        two_qubit_layers=[layers[0], layers[1]],
        lengths=lengths,
        backend=backend,
        num_samples=num_samples,
        seed=seed,
        two_qubit_gate=twoq_gate,
        one_qubit_basis_gates=oneq_gates,
    )
    lf_v = LayerFidelity(
        physical_qubits=v_qubits,
        two_qubit_layers=[layers[2], layers[3]],
        lengths=lengths,
        backend=backend,
        num_samples=num_samples,
        seed=seed,
        two_qubit_gate=twoq_gate,
        one_qubit_basis_gates=oneq_gates,
    )

    lf_h.experiment_options.max_circuits = max_circuits
    lf_v.experiment_options.max_circuits = max_circuits

    return lf_h, lf_v, layers


def _build_t1_experiments(
    backend: IBMBackend,
    delays: List[float],
) -> ParallelExperiment:
    """Create T1 experiments on all qubits in parallel."""
    qubits = list(range(backend.num_qubits))
    t1_exp = ParallelExperiment(
        [
            T1(
                physical_qubits=[q],
                delays=delays,
            )
            for q in qubits
        ],
        backend=backend,
        analysis=None,
        flatten_results=True,
    )
    return t1_exp


def _build_t2_experiments(
    backend: IBMBackend,
    delays: List[float],
) -> ParallelExperiment:
    """Create T2-Hahn experiments on all qubits in parallel."""
    qubits = list(range(backend.num_qubits))
    t2_exp = ParallelExperiment(
        [
            T2Hahn(
                physical_qubits=[q],
                delays=delays,
            )
            for q in qubits
        ],
        backend=backend,
        analysis=None,
        flatten_results=True,
    )
    return t2_exp


def _get_readout_errors(num_qubits: int, readout_result) -> Dict[int, float]:
    """Get readout errors per qubit from a job result."""
    ro_error: Dict[int, float] = {}
    cts_spam0 = readout_result[0].data.meas.get_counts()
    cts_spam1 = readout_result[1].data.meas.get_counts()
    spam_shots = readout_result[0].data.meas.num_shots
    for q in range(num_qubits):
        try:
            ro_error[q] = 1 - ((mcts(cts_spam0, [q])["0"] + mcts(cts_spam1, [q])["1"]) / 2) / spam_shots
        except KeyError:
            ro_error[q] = 1 - (mcts(cts_spam0, [q])["0"] / 2) / spam_shots
    return ro_error


def _get_oneq_properties(job_results_df: DataFrame) -> Dict[Union[int, Tuple[int, int]], float]:
    """Extract 1Q properties (1Q error, T1, T2) from a job result and return a
    corresponding dictionary."""
    values_map: Dict[Union[int, Tuple[int, int]], float] = {}
    for _, row in job_results_df.iterrows():
        key = row.components[0].index
        values_map[key] = row.value.nominal_value
    return values_map


def _get_twoq_errors(
    backend: IBMBackend,
    lf_result_h_df: DataFrame,
    lf_result_v_df: DataFrame,
    layers: List[List[Tuple[int, int]]],
) -> Dict[str, float]:
    """Extract 2Q errors from layer fidelity experiments (horiztonal and vertical) and return a
    corresponding error dictionary."""
    twoq_gate = _get_twoq_gate(backend)
    lf_err_dict = lfu.make_error_dict(backend, twoq_gate)

    updated_err_dicts = []
    for i, lf_df in enumerate([lf_result_h_df, lf_result_v_df]):
        for j in range(2):
            updated_err_dicts.append(lfu.df_to_error_dict(lf_df, layers[2 * i + j]))

    lf_err_dict = lfu.update_error_dict(lf_err_dict, updated_err_dicts)
    return lf_err_dict


def _update_qubit_props(props: Dict, prop_name: str, values_map: Dict[int, float]) -> Dict:
    """
    Update a per-qubit property in props["qubits"].

    Examples:
        props = _update_qubit_props(props, "readout_error", ro_error_map)
        props = _update_qubit_props(props, "T1", t1_map)
        props = _update_qubit_props(props, "T2", t2_map)
    """
    for q, val in values_map.items():
        for param in props["qubits"][q]:
            if param["name"] == prop_name:
                if prop_name in ["T1", "T2"]:
                    param["value"] = float(val) * 10**6 # convert to micro seconds unit
                else:
                    param["value"] = float(val)
                break
    return props


def _update_1q_errors(props: Dict, error_map: Dict[Union[int, Tuple[int, int]], float], prop_name='x') -> Dict:
    """Update 1q gate_error for sx/x from EPG maps."""
    for ix, gate in enumerate(props["gates"]):
        gate_type, component = gate['gate'], int(gate['qubits'][0])

        if gate_type == prop_name:
            for iy, parameter in enumerate(gate["parameters"]):
                if parameter["name"] == "gate_error" and component in error_map:
                    gate_error = error_map[component]
                    props["gates"][ix]["parameters"][iy]["value"] = gate_error
                    break
    return props


def _update_lf_errors(props: Dict, error_map: Dict[str, float]) -> Dict:
    """Update 2q gate_error for {cz,cx,ecr} from LF pair->error map."""
    for ix, gate in enumerate(props["gates"]):
        if gate["gate"] in ["cz", "cx", "ecr"]:
            q0, q1 = gate["qubits"]
            pair_key = f"{q0}_{q1}"
            pair_key_rev = f"{q1}_{q0}"
            if pair_key in error_map or pair_key_rev in error_map:
                gate_error = error_map.get(pair_key, error_map.get(pair_key_rev))
                for iy, param in enumerate(gate["parameters"]):
                    if param["name"] == "gate_error":
                        props["gates"][ix]["parameters"][iy]["value"] = float(gate_error)
                        break
    return props


# --------------------------------- Main Function -------------------------------- #
def characterize_backend(
    backend: IBMBackend,
    experiments: Optional[Iterable[str]] = None,
    shots: Optional[Dict[str, int]] = None,
    verbose: bool = True,
):
    """
    Run selected characterization experiments on the given IBM Quantum backend and update its properties.

    Preserved behavior (unchanged from your code):
      - Experiments:
          * "readout": spam0/spam1 over all qubits
          * "rb_1q": StandardRB per qubit, batched via Parallel+BatchExperiment
          * "rb_2q": LayerFidelity on horizontal & vertical grid layers
          * "t1": T1 on all qubits in parallel
          * "t2": T2Hahn on all qubits in parallel
      - Shots defaults: readout=10000, rb_1q=250, rb_2q=250, t1=250, t2=250
      - RB(1q): lengths=[1,50,100,500,1000,3000], num_samples=6, seed=42,
                separate_jobs=True, max_circuits=3 * num_samples
      - LF(2q): lengths=[1,10,20,30,40,60,80,100,150,200,400], num_samples=12,
                seed=60, max_circuits=144
      - T1/T2 delays: [1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6]
      - Two-qubit gate preference: ecr -> cz -> cx
      - Updates 'readout_error', 1q 'sx'/'x' gate_error (EPG), and 2q gate_error from LF,
        and patches qubit properties 'T1' and 'T2'.
      - Properties are patched on the provided backend (private attribute _properties)

    Args:
        backend: IBM Quantum backend compatible with qiskit-ibm-runtime SamplerV2.
        experiments: Optional subset of {"readout", "rb_1q", "rb_2q", "t1", "t2"}; defaults to all if None.
        shots: Optional dict overriding shots, keys in {"readout","rb_1q","rb_2q","t1","t2"}.

    Returns:
        The same backend instance with its BackendProperties updated in-place.
    """

    # ---- Local configuration  ----
    allowed_experiments = {"readout", "rb_1q", "rb_2q", "t1", "t2"}

    rb1q_lengths = np.array([1, 50, 100, 500, 1000, 3000])
    rb1q_num_samples = 6
    rb1q_seed = 42
    rb1q_samples_m = 3  # => max_circuits = 18

    lf_lengths = [1, 10, 20, 30, 40, 60, 80, 100, 150, 200, 400]
    lf_num_samples = 12
    lf_seed = 60
    lf_max_circuits = 144

    t1_delays = [1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6]
    t2_delays = [1e-6, 20e-6, 40e-6, 80e-6, 200e-6, 400e-6]

    default_shots = {"readout": 10_000, "rb_1q": 250, "rb_2q": 250, "t1": 300, "t2": 300}
    shots = {**default_shots, **(shots or {})}

    chosen = set(experiments) if experiments is not None else allowed_experiments
    unknown = chosen - allowed_experiments
    if unknown:
        raise ValueError(f"Unsupported experiments: {sorted(unknown)}. Allowed: {allowed_experiments}")

    # Print a message
    def _emit(msg: str):
        if verbose:
            print(msg, flush=True)

    # ---- Build characterization experiments ----
    sampler = Sampler(mode=backend)

    readout_circuits: Optional[List[QuantumCircuit]] = None
    if "readout" in chosen:
        readout_circuits = _build_readout_circuits(backend.num_qubits)

    sqrb_exp: Optional[BatchExperiment] = None
    if "rb_1q" in chosen:
        sqrb_exp = _build_oneq_rb_experiments(
            backend,
            lengths=rb1q_lengths,
            num_samples=rb1q_num_samples,
            seed=rb1q_seed,
            samples_m=rb1q_samples_m,
        )

    lf_h = lf_v = None
    layers: Optional[List[List[Tuple[int, int]]]] = None
    if "rb_2q" in chosen:
        lf_h, lf_v, layers = _build_layered_rb_experiments(
            backend,
            lengths=lf_lengths,
            num_samples=lf_num_samples,
            seed=lf_seed,
            max_circuits=lf_max_circuits,
        )

    t1_exp: Optional[ParallelExperiment] = None
    if "t1" in chosen:
        t1_exp = _build_t1_experiments(backend, delays=t1_delays)

    t2_exp: Optional[ParallelExperiment] = None
    if "t2" in chosen:
        t2_exp = _build_t2_experiments(backend, delays=t2_delays)

    # ---- Run and collect results ----
    if readout_circuits:
        sampler.options.default_shots = int(shots["readout"])
        job_readout = sampler.run(readout_circuits)
        try:
            _emit(f"Readout job submitted: job_id = {job_readout.job_id()}")
        except Exception:
            _emit("Readout job submitted: job_id = None")

    if sqrb_exp:
        sampler.options.default_shots = int(shots["rb_1q"])
        job_sqrb = sqrb_exp.run(sampler=sampler)
        job_ids = getattr(job_sqrb, "job_ids", None)
        _emit(f"Single-qubit RB submitted: job ids = {job_ids if job_ids is not None else 'None'}")

    if lf_h and lf_v:
        sampler.options.default_shots = int(shots["rb_2q"])
        job_lf_v = lf_v.run(sampler=sampler)
        job_lf_h = lf_h.run(sampler=sampler)

        job_ids_v = getattr(job_lf_v, "job_ids", None)
        job_ids_h = getattr(job_lf_h, "job_ids", None)

        merged_ids = []
        if job_ids_v is not None:
            merged_ids.extend(job_ids_v if isinstance(job_ids_v, list) else [job_ids_v])
        if job_ids_h is not None:
            merged_ids.extend(job_ids_h if isinstance(job_ids_h, list) else [job_ids_h])
        _emit(f"Layered two-qubit RB submitted: job_ids = {merged_ids if merged_ids else 'None'}")

    if t1_exp:
        sampler.options.default_shots = int(shots["t1"])
        job_t1 = t1_exp.run(sampler=sampler)
        job_ids_t1 = getattr(job_t1, "job_ids", None)
        _emit(f"T1 submitted: job ids = {job_ids_t1 if job_ids_t1 is not None else 'None'}")

    if t2_exp:
        sampler.options.default_shots = int(shots["t2"])
        job_t2 = t2_exp.run(sampler=sampler)
        job_ids_t2 = getattr(job_t2, "job_ids", None)
        _emit(f"T2 (Hahn) submitted: job ids = {job_ids_t2 if job_ids_t2 is not None else 'None'}")

    # ---- Build error / property maps ----
    ro_error_map: Optional[Dict[int, float]] = None
    if readout_circuits:
        readout_result = job_readout.result()
        ro_error_map = _get_readout_errors(backend.num_qubits, readout_result)

    oneq_err_x = oneq_err_sx = None
    if sqrb_exp:
        sqrb_result_x = job_sqrb.analysis_results("EPG_x", dataframe=True)
        oneq_err_x = _get_oneq_properties(sqrb_result_x)
        sqrb_result_sx = job_sqrb.analysis_results("EPG_sx", dataframe=True)
        oneq_err_sx = _get_oneq_properties(sqrb_result_sx)

    lf_err_map: Optional[Dict[str, float]] = None
    if lf_h and lf_v:
        lf_result_v = job_lf_v.analysis_results("ProcessFidelity", dataframe=True)
        lf_result_h = job_lf_h.analysis_results("ProcessFidelity", dataframe=True)
        lf_err_map = _get_twoq_errors(backend, lf_result_h, lf_result_v, layers)

    t1_map: Optional[Dict[int, float]] = None
    if t1_exp:
        t1_df = job_t1.analysis_results(dataframe=True)
        t1_map = _get_oneq_properties(t1_df)

    t2_map: Optional[Dict[int, float]] = None
    if t2_exp:
        t2_df = job_t2.analysis_results(dataframe=True)
        t2_map = _get_oneq_properties(t2_df)

    # ---- Update backend properties ----
    backend = copy.deepcopy(backend)
    props_dict = backend.properties().to_dict()
    updated_sections = []

    if ro_error_map is not None:
        props_dict = _update_qubit_props(props_dict, "readout_error", ro_error_map)
        updated_sections.append("readout")

    if oneq_err_x is not None or oneq_err_sx is not None:
        if oneq_err_x is not None:
            props_dict = _update_1q_errors(props_dict, oneq_err_x, prop_name='x')
        if oneq_err_sx is not None:
            props_dict = _update_1q_errors(props_dict, oneq_err_sx, prop_name='sx')
        updated_sections.append("single-qubit errors")

    if lf_err_map is not None:
        props_dict = _update_lf_errors(props_dict, lf_err_map)
        updated_sections.append("two-qubit errors")

    if t1_map is not None:
        props_dict = _update_qubit_props(props_dict, "T1", t1_map)
        updated_sections.append("T1")

    if t2_map is not None:
        props_dict = _update_qubit_props(props_dict, "T2", t2_map)
        updated_sections.append("T2")

    if updated_sections:
        _emit(f"Updated backend properties: {', '.join(updated_sections)}")

    props = BackendProperties.from_dict(props_dict)
    backend._properties = props  # intentional: preserved behavior

    return backend


# ----------------------------- Plotting Functions --------------------------------- #

def _extract_qubit_property(props_dict: Dict, prop_name: str) -> Dict[int, float]:
    """
    Generic extractor for per-qubit properties stored in props_dict["qubits"].

    Returns {qubit_index: value} for a given property name (e.g., "T1", "T2", "readout_error").
    """
    out = {}
    for q, q_params in enumerate(props_dict["qubits"]):
        for p in q_params:
            if p.get("name") == prop_name:
                out[q] = p.get("value")
                break
    return out


def _extract_gate_errors(
    props_dict: Dict,
    num_qubits: int,
    gate_names: List[str],
) -> Dict:
    """
    Extract gate_error values from props_dict["gates"].

    Returns:
        num_qubits == 1  -> {q: error}
        num_qubits == 2  -> {(q0, q1): error}
    """
    out = {}

    for g in props_dict["gates"]:
        if len(g["qubits"]) != num_qubits:
            continue

        gate = g["gate"]
        if gate not in gate_names:
            continue

        val = None
        for p in g["parameters"]:
            if p["name"] == "gate_error":
                val = p["value"]
                break
        if val is None:
            continue

        key = g["qubits"][0] if num_qubits == 1 else tuple(g["qubits"])
        out[key] = val

    return out


def _extract_1q_sx_errors(props_dict: Dict) -> Dict:
    return _extract_gate_errors(props_dict, num_qubits=1, gate_names=["sx"])


def _extract_1q_x_errors(props_dict: Dict) -> Dict:
    return _extract_gate_errors(props_dict, num_qubits=1, gate_names=["x"])


def _extract_twoq_gate_errors(props_dict: Dict) -> Dict:
    return _extract_gate_errors(props_dict, num_qubits=2, gate_names=["cx", "cz", "ecr"])


def _sorted_keys_by_new(old_map: Dict, new_map: Dict):
    """
    Return union of keys sorted by measured (new_map) value.
    NaNs are placed at the end.
    """
    keys = sorted(set(old_map.keys()) | set(new_map.keys()))
    return sorted(
        keys,
        key=lambda k: (np.isnan(new_map.get(k, np.nan)), new_map.get(k, np.nan)),
    )


def _set_every_xtick_with_vertical_guides(ax, labels, *, rotation=90, fontsize=7, weight="regular"):
    n = len(labels)
    idxs = np.arange(n)
    ax.set_xticks(idxs)
    ax.set_xticklabels(labels)

    for t in ax.get_xticklabels():
        t.set_rotation(rotation)
        t.set_fontsize(fontsize)
        t.set_fontweight(weight)
        if rotation == 90:
            t.set_horizontalalignment("center")
            t.set_verticalalignment("top")
        elif rotation == 45:
            t.set_horizontalalignment("right")
            t.set_verticalalignment("top")
        else:
            t.set_horizontalalignment("center")

    ax.tick_params(axis="x", which="both", width=0.6, length=3)

    for i in idxs:
        ax.axvline(i, color="k", alpha=0.08, linewidth=1.0, zorder=0)

    fig = ax.get_figure()
    fig.subplots_adjust(bottom=0.28 if rotation == 90 else 0.20)


def _plot_lines(ax, x_idx, y_old, y_new, *, lw=1.1, ms=3.0):
    ax.plot(x_idx, y_old, color="darkorange", marker="o", linewidth=lw, markersize=ms, label="Reported")
    ax.plot(x_idx, y_new, color="royalblue", marker="o", linewidth=lw, markersize=ms, label="Measured real-time")


def plot_characterization_comparison(
    old_props: Dict,
    new_props: Dict,
    plots: Iterable[str],
    title_prefix=None,
    log_scale: Optional[Dict[str, bool]] = None,
):
    """
    Plot measured vs reported comparisons for selected characterization results.

    plots:
      "readout", "rb_1q_sx", "rb_1q_x", "rb_2q", "t1", "t2"
    """
    prefix = (title_prefix + " – ") if title_prefix else ""

    use_log = {
        "readout": False,
        "rb_1q_sx": True,
        "rb_1q_x": True,
        "rb_2q": True,
        "t1": True,
        "t2": True,
    }
    if log_scale:
        use_log.update(log_scale)

    plot_configs = {
        "readout": {
            "extract": lambda props: _extract_qubit_property(props, "readout_error"),
            "title": "Readout Error — measured vs reported",
            "xlabel": "Qubit",
            "ylabel": "Readout error",
            "figsize": (18, 5.5),
            "labels": lambda keys: [str(k) for k in keys],
        },
        "rb_1q_sx": {
            "extract": _extract_1q_sx_errors,
            "title": "1Q RB EPG (sx) — measured vs reported",
            "xlabel": "Qubit",
            "ylabel": "EPG (sx)",
            "figsize": (18, 5.5),
            "labels": lambda keys: [str(k) for k in keys],
        },
        "rb_1q_x": {
            "extract": _extract_1q_x_errors,
            "title": "1Q RB EPG (x) — measured vs reported",
            "xlabel": "Qubit",
            "ylabel": "EPG (x)",
            "figsize": (18, 5.5),
            "labels": lambda keys: [str(k) for k in keys],
        },
        "rb_2q": {
            "extract": _extract_twoq_gate_errors,
            "title": "Layered 2Q RB EPG — measured vs reported",
            "xlabel": "2Q gate",
            "ylabel": "EPG (2Q)",
            "figsize": (22, 6.5),
            "labels": lambda keys: [f"{i}-{j}" for (i, j) in keys],
        },
        "t1": {
            "extract": lambda props: _extract_qubit_property(props, "T1"),
            "title": "T1 — measured vs reported",
            "xlabel": "Qubit",
            "ylabel": "T1 (s)",
            "figsize": (18, 5.5),
            "labels": lambda keys: [str(k) for k in keys],
        },
        "t2": {
            "extract": lambda props: _extract_qubit_property(props, "T2"),
            "title": "T2 (Hahn) — measured vs reported",
            "xlabel": "Qubit",
            "ylabel": "T2 (s)",
            "figsize": (18, 5.5),
            "labels": lambda keys: [str(k) for k in keys],
        },
    }

    if 'rb_1q' in plots:
        plots.append('rb_1q_x')
        plots.append('rb_1q_sx')
        plots.remove('rb_1q')

    for name in plots:
        config = plot_configs[name]

        old_map = config["extract"](old_props)
        new_map = config["extract"](new_props)

        keys = _sorted_keys_by_new(old_map, new_map)
        if not keys:
            continue

        y_old = [old_map.get(k, np.nan) for k in keys]
        y_new = [new_map.get(k, np.nan) for k in keys]
        x = np.arange(len(keys))

        fig, ax = plt.subplots(figsize=config["figsize"])
        ax.set_title(f"{prefix}{config['title']} (sorted by measured)")
        _plot_lines(ax, x, y_old, y_new, lw=1.1, ms=3.0)

        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel(config["ylabel"])

        if use_log[name]:
            ax.set_yscale("log")

        _set_every_xtick_with_vertical_guides(
            ax,
            config["labels"](keys),
            rotation=90,
            fontsize=7,
        )

        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        plt.show()
