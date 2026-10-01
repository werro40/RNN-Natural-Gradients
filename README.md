# RNN-Natural-Gradients

> Variational Monte Carlo with recurrent neural network wavefunctions — a benchmark of **minSR** (minimum-step stochastic reconfiguration) against **Adam** for variational quantum state optimization.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
  - [Key Features](#key-features)
  - [Built With](#built-with)
- [Physics Background](#-physics-background)
  - [Hamiltonians](#hamiltonians)
  - [Wavefunction Ansatz](#wavefunction-ansatz)
  - [Optimization](#optimization)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Script Reference](#script-reference)
  - [Examples](#examples)
- [Configuration](#%EF%B8%8F-configuration)
- [Outputs](#-outputs)
- [Implementation Notes](#-implementation-notes)
- [References](#-references)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

---

## 🧐 About the Project

This repository reproduces and extends a comparison of two optimization schemes for
**neural-network quantum states** (NQS) [4]: classical **Adam** and the
**minimum-step stochastic reconfiguration** algorithm (**minSR**) [2]. Ground-state
searches are carried out with Variational Monte Carlo (VMC) using recurrent neural
network (RNN) wavefunctions [1] on four model Hamiltonians:

1. **1D Transverse-Field Ising Model** (TFIM)
2. **1D Cluster state Hamiltonian**
3. **2D Heisenberg model** (square lattice, optional C4v symmetrization)
4. **2D frustrated J1–J2 model** (J2/J1 = 0.5)

### Key Features

- 🚀 **Two optimizers, one framework**: Adam [4] and minSR [2] natural-gradient steps share the same sampling, local-energy, and logging machinery.
- ⚡ **Memory-aware Jacobians**: energy-gradient Jacobians are computed and flattened in sequential batches (`jax.lax.map`), allowing large sample counts on a single accelerator.
- 🛡️ **Levenberg–Marquardt damping**: the minSR normal equations are regularized, with optional trust-region step-size control for the frustrated model.
- 🧩 **Symmetry augmentation**: C4v point-group symmetrization of the 2D wavefunction via log-sum-exp averaging over the eight group elements [3].
- 📈 **Reproducible logging**: every run records energies, variances, wall-clock times, metadata, plots (PDF), model checkpoints (pickle), and a plain-text experiment description.

### Built With

- [Python](https://www.python.org/) — core language (Python 3.10–3.13, as required by the pinned JAX version)
- [JAX](https://github.com/jax-ml/jax) — vectorized, JIT-compiled autodiff backend
- [Flax](https://github.com/google/flax) — neural network modules (RNN cells, Dense layers)
- [Optax](https://github.com/google-deepmind/optax) — gradient-based optimizers (SGD + momentum, Adam)
- [NumPy](https://numpy.org/) / [Matplotlib](https://matplotlib.org/) — data handling and plotting
- [Optuna](https://optuna.org/) — hyperparameter search support
- [PyYAML](https://pyyaml.org/) — configuration files
- [tqdm](https://tqdm.github.io/) — progress bars

---

## ⚛️ Physics Background

All simulations target the ground state of a lattice Hamiltonian `H` by minimizing the
variational energy

```text
E(θ) = ⟨Ψθ|H|Ψθ⟩ / ⟨Ψθ|Ψθ⟩ = ⟨E_loc⟩_{s ~ |Ψθ|²}
```

where `E_loc(s) = Σ_{s'} H_{s's} Ψθ(s')/Ψθ(s)` is the **local energy** of a spin
configuration `s` drawn from the Born distribution of the wavefunction.

### Hamiltonians

Spin configurations are stored as integers `s_i ∈ {0, 1}` and mapped to Pauli-z
eigenvalues `σ_i = 2 s_i − 1 ∈ {−1, +1}`. All models use open boundary conditions.

| Script | Hamiltonian | Terms |
| --- | --- | --- |
| `TFIM-1D.py` | 1D TFIM | `H = −Σᵢ σᶻᵢσᶻᵢ₊₁ − h Σᵢ σˣᵢ` |
| `Cluster-1D.py` | 1D cluster state | stabilizer Hamiltonian `H = −Σᵢ σˣᵢ₋₁σᶻᵢσˣᵢ₊₁` (with OBC edge corrections) |
| `Heisenberg-2D.py` | 2D Heisenberg | `H = Σ_{⟨i,j⟩} Sᵢ·Sⱼ` on an `N×N` square lattice |
| `J1J2-2D.py` | 2D J1–J2 | `H = J1 Σ_{⟨i,j⟩} Sᵢ·Sⱼ + J2 Σ_{⟨⟨i,j⟩⟩} Sᵢ·Sⱼ`, `J2/J1 = 0.5` |


The local energies (`Utils/utils.py`) are computed by enumerating all spin
configurations connected to `s` by a single off-diagonal Hamiltonian term (spin
flips on coupled bonds), re-evaluating the network on each connected configuration
inside a `jax.lax.fori_loop`, and accumulating `H_{s's} Ψ(s')/Ψ(s)`.

### Wavefunction Ansatz

- **1D models** (`CRNNModel`, `RNNModel` in `Utils/models.py`): an autoregressive RNN
  (GRU / LSTM / Vanilla cell) that processes the chain site by site and emits
  conditional probabilities for each spin. The complex variant adds a phase head,
  `φ(s) = π · softsign(W x)`, giving
  `ln Ψθ(s) = ½ log pθ(s) + i φ(s)`.
- **2D models** (`StackedPRNNModel`, `StackedCRNNModel`): stacked 2D RNN layers
  (each cell receives hidden states from the left and upper neighbors) are applied
  along a **serpentine (zigzag) path** that sweeps the lattice column by column;
  outputs pass through gated linear units (GLU). Sampling and evaluation are exact
  autoregressive procedures, so no Markov-chain equilibration is required.
- **Symmetry augmentation** (`StackedPRNNModel.logprobs_c4vsym`): the log-amplitude
  of a configuration is averaged over the eight C4v image configurations
  (four rotations, four reflections) via
  `ln (1/|G| Σ_g |Ψ(g s)|²)` using `logsumexp` [3].

### Optimization

Given the Jacobian `X_{sα} = ∂ ln Ψθ(s) / ∂θ_α` (with real and imaginary parts
concatenated for complex wavefunctions), minSR [2] replaces the full `O(P²)` Fisher
matrix of stochastic reconfiguration [5] with the `O(ns²)` normal equations

```text
( X X† + λ I ) x = f        (Levenberg–Marquardt damped, solved via Cholesky)
Δθ = X† x
```

where `f` is the centered, `√ns`-normalized force vector built from the local
energies. The update is applied through an `optax` optimizer (SGD with momentum and
an inverse-time learning-rate decay `lr/(1 + step/step_decay)`). Adam runs use the
standard backpropagated VMC gradient estimator `2 Re[⟨(E_loc − ⟨E_loc⟩) ∂ ln Ψθ*⟩]`.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10–3.13** (the pinned `jax==0.7.2` wheels cover these versions)
- A working JAX backend: CPU works out of the box; for GPU support install the
  matching `jax[cuda]` wheels — see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
- ~1 GB of disk space per production run for data, plots, and checkpoints.

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd RNN-Natural-Gradients

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Install the pinned dependencies
pip install -r requirements.txt
```

The scripts also expect a directory for the initial network parameters:

```bash
mkdir -p init-params
```

`Heisenberg-2D.py` automatically generates and saves `init-params/h2d_params-<dh>.pkl`
on first use; `TFIM-1D.py` and `J1J2-2D.py` load previously saved pickles from the
same directory (see [Script Reference](#script-reference)).

---

## 🧪 Usage

### Script Reference

All entry points live in the root directory and share `Utils/`. Every run creates a
timestamped experiment folder (see [Outputs](#-outputs)).

| Script | Hamiltonian | Wavefunction | Optimizer(s) | Configuration source |
| --- | --- | --- | --- | --- |
| `TFIM-1D.py` | 1D TFIM (N=200) | `RNNModel` (GRU) | Adam (minSR branch included) | constants at the top of the script (`opt_dict`, `N`, `dh`, …); loads `init-params/tfim_params-32.pkl` |
| `Cluster-1D.py` | 1D cluster (N=30) | `CRNNModel` (complex, GRU) | minSR | constants at the top of the script; optional positional task id |
| `Heisenberg-2D.py` | 2D Heisenberg (N×N) | `StackedPRNNModel` (GRU) | minSR | **YAML config file** (required) + CLI flags |
| `J1J2-2D.py` | 2D J1–J2 (6×6) | stacked 2D RNN (complex, GRU) | minSR with trust region | constants at the top of the script; positional key indexes the sample-count list `nss` |

CLI arguments:

```text
python Heisenberg-2D.py  <key> [-t] [-grid] --config PATH [--time_limit D-HH:MM:SS]
python J1J2-2D.py        <key> [-t]
python Cluster-1D.py     [task_id]
```

- `<key>` — positional seed/index. In `Heisenberg-2D.py` it seeds model
  initialization and names checkpoint files; in `J1J2-2D.py` it selects the number
  of Monte Carlo samples from the study list `nss = [20, 80, 120, …, 1240]`.
- `-t, --test` — smoke-test mode (tiny lattice, a few steps) to verify the
  installation end to end.
- `-grid` — accepted by `Heisenberg-2D.py` (grid-GRU variant switch).
- `--time_limit` — Slurm-style wall-clock limit (`D-HH:MM:SS`, `HH:MM:SS`, or
  `MM:SS`); `Heisenberg-2D.py` stops training gracefully before this deadline.

### Examples

```bash
# 1D cluster state with a complex RNN (minSR); default task id 0
python Cluster-1D.py 0

# 2D Heisenberg driven entirely by its YAML configuration
python Heisenberg-2D.py 0 --config Configs/config_h2d.yaml --time_limit 2-10:00:00

# Quick smoke test of the 2D Heisenberg pipeline (seconds, CPU-friendly)
python Heisenberg-2D.py 0 -t --config Configs/config_h2d.yaml

# 1D TFIM with Adam; hyperparameters are the constants at the top of the file
python TFIM-1D.py

# 2D J1-J2 model with a complex 2D RNN (minSR or Adam)
python J1J2-2D.py 0 --config Configs/config_j1j2.yaml
```

---

## ⚙️ Configuration

`Configs/config_h2d.yaml` is the runtime configuration consumed by
`Heisenberg-2D.py`; it is copied verbatim into the experiment folder so
every run is self-documenting. The remaining files (`config_tfim.yaml`,
`config_cluster.yaml`, `config_j1j2.yaml`) are reference records of the values used
by those scripts, which currently take their parameters from constants defined at
the top of each file.

| Key | Type | Meaning |
| --- | --- | --- |
| `title` | str | Label used in experiment folder names |
| `opt` | str | `minsr` or `adam` |
| `N` | int | Linear lattice size (`N×N` sites for 2D models) |
| `dh` | int | Hidden dimension of the RNN cells |
| `batches` | int | Number of sequential Jacobian batches (memory control) |
| `samples_per_batch` | int | Monte Carlo samples per batch (`ns = batches × samples_per_batch`) |
| `steps` | int | Number of optimization steps |
| `lr` | float | Base learning rate |
| `m` | float | SGD momentum |
| `step_decay` | int | Inverse-time decay: `lr_t = lr / (1 + t/step_decay)` |
| `lambda_reg` | float | Initial Levenberg–Marquardt damping λ [unused by Adam] |
| `max_lambda` | float | Upper bound for adaptive damping [unused by Adam] |
| `lambda_factor` | float | Multiplicative λ growth factor [unused by Adam] |
| `numsamples_check` | int | Sample count for periodic checkpoint evaluations |
| `numsamples_final` | int | Sample count for the final high-precision energy estimate |

---

## 📊 Outputs

Each run writes a self-contained experiment directory:

```text
Experiments/
├── <OPT>/TFIM-1D-<date>-<title>/        # from TFIM-1D.py
├── <OPT>/Cluster-1D-<date>-<title>/     # from Cluster-1D.py
├── <OPT>/2DH-<date>-<title>/            # from Heisenberg-2D.py
└── <OPT>/J1J2-2D-<date>-<title>/        # from J1J2-2D.py
    ├── Data/       energies / variances / times (.npz)
    ├── Plots/      metric histories (.pdf)
    ├── Models/     parameter checkpoints (.pkl)
    ├── config.yaml snapshot of the runtime configuration
    └── docs.txt    plain-text description of the experiment setup
```

- `Data/*.npz` archives produced by `data_class.save` bundle the tracked metric
  series together with a metadata dictionary (optimizer, date, per-site energy and
  error, sample counts).
- The final energy and statistical error are obtained from independent evaluation
  batches with an online mean/variance accumulator (`final_sampler` /
  `final_sampler_maker`), so the quoted error bars are Monte Carlo standard errors.

---

## 🔧 Implementation Notes

- **Double precision everywhere**: `jax.config.update("jax_enable_x64", True)` is set
  in every module; all parameters are `float64`, and complex wavefunctions use
  `complex128`.
- **Exact autoregressive sampling**: configurations are generated site by site from
  the conditional distributions, so estimators are reweighting-free.
- **Batched Jacobians**: `jax.lax.map` evaluates the Jacobian of
  `ln Ψθ` over sample batches sequentially (instead of `vmap`), trading a small
  amount of speed for bounded memory on large `ns`.
- **Jacobian hygiene**: per-parameter blocks are centered (`X ← X − ⟨X⟩_s`) and
  scaled by `1/√ns` before forming the normal matrix `X X†`, and the update
  direction is normalized by `‖τ‖` (`make_training_step` in `Utils/utils.py`).
- **Complex SR**: for complex wavefunctions the real and imaginary Jacobians are
  stacked into `X = [Re X; Im X]` and the force into `f = [Re f; −Im f]`, and the
  damped system is solved with a Cholesky factorization
  (`jax.scipy.linalg.cho_factor/cho_solve`).
- **Watchdogs**: training loops abort on `NaN` energies and on wall-clock limits;
  `Heisenberg-2D.py` parses `--time_limit` (including Slurm `D-HH:MM:SS` strings)
  for cluster integration.
- **JIT compilation**: sampling, local-energy evaluation, and full optimization
  steps are `jax.jit`-compiled; the first iteration of every run includes compile
  time and is slower than subsequent ones.

---

## 📚 References

1. M. Hibat-Allah, M. Ganahl, L. E. Hayward, R. G. Melko, and J. Carrasquilla,
   *Recurrent neural network wave functions*,
   [Phys. Rev. Research **2**, 023358 (2020)](https://link.aps.org/doi/10.1103/PhysRevResearch.2.023358),
   [arXiv:2002.02973](https://arxiv.org/abs/2002.02973).
2. A. Chen and M. Heyl, *Empowering deep neural quantum states through efficient
   optimization*, [Nat. Phys. **20**, 1176 (2024)](https://www.nature.com/articles/s41567-024-02566-1),
   [arXiv:2302.01941](https://arxiv.org/abs/2302.01941).
3. M. Hibat-Allah, M. Mauri, J. Carrasquilla, and A. G. Ferreira,
   *Supplementing recurrent neural network wave functions with symmetry and
   annealing*, [arXiv:2207.14314](https://arxiv.org/abs/2207.14314).
4. G. Carleo and M. Troyer, *Solving the quantum many-body problem with artificial
   neural networks*, [Science **355**, 602 (2017)](https://www.science.org/doi/10.1126/science.aag2302).
5. S. Sorella, *Wave function optimization in the variational Monte Carlo method*,
   [Phys. Rev. B **64**, 024512 (2001)](https://link.aps.org/doi/10.1103/PhysRevB.64.024512).

---

## 🤝 Contributing

Contributions are welcome — bug reports, documentation improvements, and new
Hamiltonian/ansatz implementations. Please open an issue describing the proposed
change before submitting a pull request, and keep new code consistent with the
existing style: documented functions, typed signatures where practical, and
experiment outputs grouped under `Experiments/`.

---

## 📄 License

This project is distributed under a custom ethical license derived from
**BSD-3-Clause** with additional *Do-No-Harm* clauses (human-rights consistency,
no use against climate action, no military applications). See the
[LICENSE](LICENSE) file for the full text.

---

## ✉️ Contact

Maintained by **Adil Attar and contributors** — please use GitHub issues for
questions and bug reports.

---

## 🙏 Acknowledgments

- The RNN wavefunction architecture follows M. Hibat-Allah *et al.* [1]; a related
  reference implementation is available at
  [mhibatallah/RNNWavefunctions](https://github.com/mhibatallah/RNNWavefunctions).
- The minSR algorithm is due to A. Chen and M. Heyl [2].
- Thanks to the JAX, Flax, and Optax developer teams for the underlying ecosystem.
- AI was used in refining and documenting the code.
