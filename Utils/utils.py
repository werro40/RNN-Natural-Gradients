"""
Utility Functions and Hamiltonian Physics Operators for Variational Monte Carlo (VMC) Calculations.

This module provides support classes and functions to run VMC simulations,
including a structured metric tracker, flat-Pytree helper functions for
Jacobian computations, local energy estimators for 1D/2D models,
sequential samplers, and training step compilers.

The main components are:

- ``data_class``: metric logging, checkpointing (.npz) and plotting (.pdf).
- ``slurm_time_to_seconds``: parsing of Slurm wall-clock limit strings.
- ``_flatten_jacobian`` / ``_unflatten_like_params``: helpers to convert the
  PyTree-structured Jacobian of the wavefunction into a dense
  (numsamples x numparams) matrix and back.
- Local energy estimators ``local_energy_*`` for the supported Hamiltonians:
  2D Heisenberg (plain and C4v-symmetrized), 2D J1-J2, 1D TFIM, and the 1D
  cluster state Hamiltonian.
- ``local_energy_generator``: Hamiltonian lookup/closure factory.
- ``make_training_step``: JIT-compiled minSR optimization step (Levenberg-
  Marquardt damped stochastic reconfiguration over batched Jacobians).
- ``final_sampler`` / ``final_sampler_maker``: high-precision evaluation of
  the converged energy with online batch merging of mean and variance.

Conventions: spins are stored as integers s_i in {0, 1} and mapped to Pauli-z
eigenvalues sigma_i = 2 s_i - 1; lattices use open boundary conditions; all
arithmetic runs in float64 / complex128.
"""

import os
import math
import time
import json
from functools import partial
from typing import List, Tuple, Union, Optional, Callable, Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

# Force x64 precision for numerical stability in physical simulations
jax.config.update("jax_enable_x64", True)
jax_dtype = jnp.float64


class data_class:
    """
    Utility class to log and visualize training metrics (Energy, Variance, Time, etc.).

    Metrics are declared once as keyword arguments at construction
    (e.g., ``data_class(Energy=[], Var=[], Time=[])``) and appended in order
    with :meth:`update`. Supported output formats are compressed ``.npz``
    archives (metrics plus a metadata dictionary) and per-metric PDF plots.
    """
    def __init__(self, **kwargs):
        """Initializes lists for each metric keyword argument."""
        self._keys = list(kwargs.keys())
        for key in self._keys:
            setattr(self, key, [])

    def update(self, args: List[Any]):
        """Appends new measurements to the tracked metric lists in defined order."""
        if len(args) != len(self._keys):
            raise ValueError(f"Expected {len(self._keys)} arguments, got {len(args)}")
        for idx, key in enumerate(self._keys):
            getattr(self, key).append(args[idx])

    def save(self, folder: str, metadata: dict = {}, opt: str = 'minsr', i: int = 0):
        """Saves metrics and run metadata in a compressed numpy archive (.npz)."""
        data_to_save = {key: jnp.array(getattr(self, key)) for key in self._keys}
        data_to_save['metadata'] = metadata
        key1 = self._keys[0]
        step = len(getattr(self, key1))
        
        # Ensure target directory exists
        os.makedirs(os.path.join(folder, 'Data'), exist_ok=True)
        jnp.savez(os.path.join(folder, f'Data/data-{opt}-s{step}-{i}.npz'), **data_to_save)

    def plot(self, key: str, folder: str, opt: str = 'minsr', i: int = 0, log: bool = False, **kwargs):
        """Plots the selected metric history and saves as a high-DPI PDF."""
        x = jnp.abs(jnp.array(getattr(self, key)))
        steps = len(x)
        
        fig, ax = plt.subplots(dpi=100)
        ax.plot(x, **kwargs)
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel(key, fontsize=16)
        ax.grid(True)
        plt.legend()
        if log:
            ax.set_yscale('log')
        plt.tight_layout()
        
        os.makedirs(os.path.join(folder, 'Plots'), exist_ok=True)
        plt.savefig(os.path.join(folder, f'Plots/{key}-{opt}-s{steps}-{i}.pdf'))
        plt.close()


def slurm_time_to_seconds(t: str) -> int:
    """
    Converts a Slurm-formatted time string to seconds.

    Accepts the formats ``D-HH:MM:SS``, ``HH:MM:SS``, and ``MM:SS``
    (the latter two are zero-padded to full ``HH:MM:SS`` as needed).

    Args:
        t: Slurm time limit string, e.g. "2-10:00:00".

    Returns:
        Total wall-clock time in seconds.
    """
    days = 0
    if '-' in t:
        days, t = t.split('-')
        days = int(days)
    parts = [int(p) for p in t.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)  # pad in case format is just MM:SS
    hours, minutes, seconds = parts
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _apply_step(params: Any, dtheta_tree: Any, step_size: float) -> Any:
    """Computes parameters updated via standard stochastic gradient update (params - step_size * dtheta)."""
    return jax.tree.map(lambda p, d: p - step_size * d, params, dtheta_tree)


def _flatten_jacobian(jacobian: Any, numsamples: int) -> Tuple[jnp.ndarray, Any, List[Tuple], List[slice]]:
    """
    Flattens a PyTree-structured Jacobian into a dense (numsamples x numparams) matrix.

    The returned metadata (tree structure, per-leaf shapes, and column slices)
    is passed to :func:`_unflatten_like_params` to map a flat update vector
    back onto the parameter PyTree.

    Args:
        jacobian: Per-parameter PyTree of arrays with a leading sample axis.
        numsamples: Number of Monte Carlo samples (rows of the matrix).

    Returns:
        Tuple of (dense Jacobian matrix, PyTree structure, leaf shapes, column
        slices).
    """
    flattened_jac, tree = jax.tree_util.tree_flatten(jacobian)
    shapes = [it.shape for it in flattened_jac]
    sizes = [it[0].size for it in flattened_jac]
    slices = []
    last = 0
    for s in sizes:
        slices.append(slice(last, last + s))
        last += s
    jac = jnp.concatenate([it.reshape(it.shape[0], -1) for it in flattened_jac], axis=-1)
    return jac, tree, shapes, slices


def _unflatten_like_params(flat_vec: jnp.ndarray, tree: Any, shapes: List[Tuple], slices: List[slice]) -> Any:
    """Inverse of _flatten_jacobian for a single flattened parameter-shaped vector."""
    flat_tree = []
    for shape, _slice in zip(shapes, slices):
        flat_tree.append(flat_vec[_slice].reshape(shape[1:]))
    return jax.tree_util.tree_unflatten(tree, flat_tree)


def local_energy_h2d(samples: jnp.ndarray, params: Any, model: Any, log_psi: jnp.ndarray) -> jnp.ndarray:
    """
    Computes local energies for the 2D Heisenberg model on a square lattice with open boundary conditions.

    H = Sum_{<i,j>} S_i . S_j

    The diagonal contribution sums the Sz*Sz products over horizontal and
    vertical nearest-neighbor bonds. Off-diagonal contributions come from the
    Sx*Sx + Sy*Sy = 0.5*(S+S- + S-S+) flip terms: for every bond whose two
    spins are anti-aligned, the network is re-evaluated on the state with that
    bond flipped and the ratio Psi(s')/Psi(s) is accumulated.

    Args:
        samples: Spin configurations of shape (numsamples, Nx, Ny) with s in {0, 1}.
        params: Model parameters (PyTree).
        model: Flax module evaluating the log-wavefunction.
        log_psi: Real log-amplitudes 0.5 * ln p(s) of the input samples.

    Returns:
        Local energies E_loc(s) of shape (numsamples,).
    """
    numsamples, Nx, Ny = samples.shape

    local_energies = jnp.zeros((numsamples), dtype=jax_dtype)
    # Diagonal components (Sz * Sz interactions)
    local_energies += jnp.sum(0.25 * (2 * samples[:, :-1, :] - 1) * (2 * samples[:, 1:, :] - 1), axis=(1, 2))
    local_energies += jnp.sum(0.25 * (2 * samples[:, :, :-1] - 1) * (2 * samples[:, :, 1:] - 1), axis=(1, 2))

    # Off-diagonal exchange terms (Sx*Sx + Sy*Sy) using sequential spin flips (autoregressive ratio check)
    def step_fn_horizontal(n, state):
        s, output = state
        i = n // Ny
        j = n % Ny

        # Flip adjacent spins horizontally
        flipped_state = s.at[:, i, j].set(1 - s[:, i, j])
        flipped_state = flipped_state.at[:, i + 1, j].set(1 - flipped_state[:, i + 1, j])
        flipped_logpsi = 0.5 * model.apply(params, flipped_state)
        # Ratio of psi(s') / psi(s)
        output += (s[:, i, j] + s[:, i + 1, j] == 1) * (-0.5) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    def step_fn_vertical(n, state):
        s, output = state
        j = n // Nx
        i = n % Nx

        # Flip adjacent spins vertically
        flipped_state = s.at[:, i, j].set(1 - s[:, i, j])
        flipped_state = flipped_state.at[:, i, j + 1].set(1 - flipped_state[:, i, j + 1])
        flipped_logpsi = 0.5 * model.apply(params, flipped_state)
        output += ((s[:, i, j] + s[:, i, j + 1] == 1) * (-0.5)) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    output = jnp.zeros((numsamples), dtype=jax_dtype)
    _, off_diag_term_vertical = jax.lax.fori_loop(0, Nx * (Ny - 1), step_fn_vertical, (samples, output))
    _, off_diag_term_horizontal = jax.lax.fori_loop(0, (Nx - 1) * Ny, step_fn_horizontal, (samples, output))

    local_energies += off_diag_term_vertical + off_diag_term_horizontal
    return local_energies


def local_energy_h2d_sym(samples: jnp.ndarray, params: Any, model: Any, log_psi: jnp.ndarray) -> jnp.ndarray:
    """
    Computes local energies for the 2D Heisenberg model using C4v symmetry-averaged amplitudes.

    Identical to :func:`local_energy_h2d` except that the wavefunction ratios
    Psi(s')/Psi(s) are evaluated with the C4v-symmetrized wavefunction
    (``model.logprobs_c4vsym``), which reduces the variance of the estimator
    when the ansatz is symmetrized.

    See :func:`local_energy_h2d` for the Hamiltonian and argument conventions.
    """
    numsamples, Nx, Ny = samples.shape

    local_energies = jnp.zeros((numsamples), dtype=jax_dtype)
    local_energies += jnp.sum(0.25 * (2 * samples[:, :-1, :] - 1) * (2 * samples[:, 1:, :] - 1), axis=(1, 2))
    local_energies += jnp.sum(0.25 * (2 * samples[:, :, :-1] - 1) * (2 * samples[:, :, 1:] - 1), axis=(1, 2))

    def step_fn_horizontal(n, state):
        s, output = state
        i = n // Ny
        j = n % Ny

        flipped_state = s.at[:, i, j].set(1 - s[:, i, j])
        flipped_state = flipped_state.at[:, i + 1, j].set(1 - flipped_state[:, i + 1, j])
        # Use sym-evaluated logprobs
        flipped_logpsi = 0.5 * model.apply(params, flipped_state, method="logprobs_c4vsym")
        output += (s[:, i, j] + s[:, i + 1, j] == 1) * (-0.5) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    def step_fn_vertical(n, state):
        s, output = state
        j = n // Nx
        i = n % Nx

        flipped_state = s.at[:, i, j].set(1 - s[:, i, j])
        flipped_state = flipped_state.at[:, i, j + 1].set(1 - flipped_state[:, i, j + 1])
        flipped_logpsi = 0.5 * model.apply(params, flipped_state, method="logprobs_c4vsym")
        output += ((s[:, i, j] + s[:, i, j + 1] == 1) * (-0.5)) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    output = jnp.zeros((numsamples), dtype=jax_dtype)
    _, off_diag_term_vertical = jax.lax.fori_loop(0, Nx * (Ny - 1), step_fn_vertical, (samples, output))
    _, off_diag_term_horizontal = jax.lax.fori_loop(0, (Nx - 1) * Ny, step_fn_horizontal, (samples, output))

    local_energies += off_diag_term_vertical + off_diag_term_horizontal
    return local_energies


def local_energy_j1j2(samples: jnp.ndarray, params: Any, model: Any, log_psi: jnp.ndarray, 
                      J1: float = 1.0, J2: float = 0.5) -> jnp.ndarray:
    """
    Computes local energies for the J1-J2 model on a square lattice with open boundary conditions.

    H = J1 Sum_{<i,j>} S_i . S_j + J2 Sum_{<<i,j>>} S_i . S_j

    The diagonal part includes J1 nearest-neighbor bonds (horizontal and
    vertical) and J2 next-nearest-neighbor bonds along both lattice diagonals.
    Off-diagonal contributions enumerate single-bond spin flips for all four
    bond types; flip terms carry a +0.5 sign for the J2 diagonal bonds
    (S+S- + S-S+ = 2(SxSx + SySy)) instead of the -0.5 used for J1 bonds,
    following the standard S+.S- decomposition of the exchange interaction.

    Args:
        samples: Spin configurations of shape (numsamples, Nx, Ny).
        params: Model parameters (PyTree).
        model: Flax module evaluating the (complex) log-wavefunction.
        log_psi: Real log-amplitudes 0.5 * ln p(s) of the input samples.
        J1: Nearest-neighbor coupling.
        J2: Next-nearest-neighbor coupling (frustration when J2/J1 > 0).

    Returns:
        Local energies E_loc(s) of shape (numsamples,), complex-valued.
    """
    numsamples, Nx, Ny = samples.shape

    local_energies = jnp.zeros((numsamples), dtype=jax_dtype)

    sigmap = 2 * samples - 1
    local_energies += 0.25 * J1 * jnp.sum(sigmap[:, :, :-1] * sigmap[:, :, 1:], axis=(1, 2))  # horizontal
    local_energies += 0.25 * J1 * jnp.sum(sigmap[:, :-1, :] * sigmap[:, 1:, :], axis=(1, 2))  # vertical
    local_energies += 0.25 * J2 * jnp.sum(sigmap[:, :-1, :-1] * sigmap[:, 1:, 1:], axis=(1, 2))  # diagonal top-left to bottom-right
    local_energies += 0.25 * J2 * jnp.sum(sigmap[:, :-1, 1:] * sigmap[:, 1:, :-1], axis=(1, 2))  # diagonal top-right to bottom-left

    def step_fn_horizontal(n, state):
        s, output = state
        i = n // Ny
        j = n % Ny

        flipped_state = s.at[:, i, j].set(1 - s[:, i, j])
        flipped_state = flipped_state.at[:, i + 1, j].set(1 - flipped_state[:, i + 1, j])
        flipped_logpsi = model.apply(params, flipped_state)
        output += (s[:, i, j] + s[:, i + 1, j] == 1) * (-0.5) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    def step_fn_right(n, state):
        s, output = state
        i = n // (Ny - 1)
        j = n % (Ny - 1)

        flipped_state = s.at[:, i, j].set(1 - s[:, i, j])
        flipped_state = flipped_state.at[:, i + 1, j + 1].set(1 - flipped_state[:, i + 1, j + 1])
        flipped_logpsi = model.apply(params, flipped_state)
        output += J2 * (s[:, i, j] + s[:, i + 1, j + 1] == 1) * (0.5) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    def step_fn_left(n, state):
        s, output = state
        i = n // (Ny - 1)
        j = n % (Ny - 1)
        j += 1

        flipped_state = s.at[:, i, j].set(1 - s[:, i, j])
        flipped_state = flipped_state.at[:, i + 1, j - 1].set(1 - flipped_state[:, i + 1, j - 1])
        flipped_logpsi = model.apply(params, flipped_state)
        output += J2 * (s[:, i, j] + s[:, i + 1, j - 1] == 1) * (0.5) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    def step_fn_vertical(n, state):
        s, output = state
        j = n // Nx
        i = n % Nx

        flipped_state = s.at[:, i, j].set(1 - s[:, i, j])
        flipped_state = flipped_state.at[:, i, j + 1].set(1 - flipped_state[:, i, j + 1])
        flipped_logpsi = model.apply(params, flipped_state)
        output += ((s[:, i, j] + s[:, i, j + 1] == 1) * (-0.5)) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    output = jnp.zeros((numsamples), dtype=jnp.complex128)
    _, off_diag_term_vertical = jax.lax.fori_loop(0, Nx * (Ny - 1), step_fn_vertical, (samples, output))
    _, off_diag_term_horizontal = jax.lax.fori_loop(0, (Nx - 1) * Ny, step_fn_horizontal, (samples, output))
    _, off_diag_term_right = jax.lax.fori_loop(0, (Nx - 1) * (Ny - 1), step_fn_right, (samples, output))
    _, off_diag_term_left = jax.lax.fori_loop(0, (Ny - 1) * (Nx - 1), step_fn_left, (samples, output))

    local_energies += off_diag_term_vertical + off_diag_term_horizontal + off_diag_term_left + off_diag_term_right
    return local_energies


def local_energy_tfim(samples: jnp.ndarray, params: Any, model: Any, log_psi: jnp.ndarray) -> jnp.ndarray:
    """
    Computes local energies for the 1D Transverse Field Ising Model (h = 1) with open boundary conditions.

    H = - Sum_{i=1}^{N-1} Z_i Z_{i+1} - h Sum_i X_i,   h = 1

    The diagonal ZZ interaction sums over N-1 nearest-neighbor bonds; the
    roll-based sum would double count the periodic image, so the boundary bond
    contribution is subtracted to enforce OBC. The off-diagonal transverse
    field is handled by flipping each spin in turn, re-evaluating the network,
    and accumulating -Psi(s_i flipped)/Psi(s).

    Args:
        samples: Spin configurations of shape (numsamples, N).
        params: Model parameters (PyTree).
        model: Flax module evaluating the log-wavefunction.
        log_psi: Real log-amplitudes 0.5 * ln p(s) of the input samples.

    Returns:
        Local energies E_loc(s) of shape (numsamples,).
    """
    numsamples, N = samples.shape

    # Diagonal nearest neighbor interaction
    interaction_term = - jnp.sum((2 * samples - 1) * (2 * jnp.roll(samples, 1, axis=1) - 1), axis=1)
    # Impose OBC boundary terms
    interaction_term += (2 * samples[:, 0] - 1) * (2 * samples[:, -1] - 1)

    # Off-diagonal transverse field updates
    def step_fn_transverse(i, state):
        s, output = state
        flipped_state = s.at[:, i].set(1 - s[:, i])
        flipped_logpsi = 0.5 * model.apply(params, flipped_state)
        output += - jnp.exp(flipped_logpsi - log_psi)
        return s, output

    output = jnp.zeros((numsamples), dtype=jnp.float64)
    _, off_diag_term = jax.lax.fori_loop(0, N, step_fn_transverse, (samples, output))

    return interaction_term + off_diag_term


def local_energy_cluster(samples: jnp.ndarray, params: Any, model: Any, log_psi: jnp.ndarray) -> jnp.ndarray:
    """
    Computes local energies for the 1D cluster state Hamiltonian with open boundary conditions.

    H = - Sum_i X_{i-1} Z_i X_{i+1}

    Because X_{i-1} Z_i X_{i+1} flips the two neighbors of site i, each
    off-diagonal contribution flips the pair (i-1, i+1) and re-evaluates the
    network; the factor -(1 - 2 s_i) accounts for the Z_i eigenvalue of the
    central spin. The bulk loop covers interior sites and the remaining
    three terms handle the open-boundary edge corrections at i = 0, i = N-1,
    and the two-site boundary term.

    Args:
        samples: Spin configurations of shape (numsamples, N).
        params: Model parameters (PyTree).
        model: Flax module evaluating the (complex) log-wavefunction.
        log_psi: Real log-amplitudes 0.5 * ln p(s) of the input samples.

    Returns:
        Local energies E_loc(s) of shape (numsamples,), complex-valued.
    """
    NUMBER_OF_SAMPLES, N = samples.shape

    def step_fn_cluster(i, state):
        s, output = state
        flipped_state = s.at[:, i - 1].set(1 - s[:, i - 1])
        flipped_state = flipped_state.at[:, i + 1].set(1 - flipped_state[:, i + 1])
        flipped_logpsi = model.apply(params, flipped_state)
        output += -(1 - 2 * flipped_state[:, i]) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    output = jnp.zeros((NUMBER_OF_SAMPLES), dtype=jnp.complex128)
    _, off_diag_term = jax.lax.fori_loop(1, N - 2, step_fn_cluster, (samples, output))

    flipped_state = samples.at[:, 1].set(1 - samples[:, 1])
    flipped_logpsi = model.apply(params, flipped_state)
    off_diag_term += -(1 - 2 * flipped_state[:, 0]) * jnp.exp(flipped_logpsi - log_psi)

    flipped_state = samples.at[:, N - 2].set(1 - samples[:, N - 2])
    flipped_state = flipped_state.at[:, N - 1].set(1 - flipped_state[:, N - 1])
    flipped_logpsi = model.apply(params, flipped_state)
    off_diag_term += -jnp.exp(flipped_logpsi - log_psi)

    flipped_state = samples.at[:, N - 3].set(1 - samples[:, N - 3])
    flipped_logpsi = model.apply(params, flipped_state)
    off_diag_term += -(1 - 2 * flipped_state[:, N - 2]) * (1 - 2 * flipped_state[:, N - 1]) * jnp.exp(flipped_logpsi - log_psi)

    return off_diag_term


def final_sampler(final_samples: int, batches: int, params: Any, model: Any, N: int) -> Tuple[float, float]:
    """
    Runs a batch-averaged evaluation sampler for the 2D Heisenberg model to obtain a precise energy estimate.

    Draws ``final_samples`` configurations in ``batches`` independent chunks,
    evaluates the local energy on each chunk, and merges the per-batch means
    and variances online (Welford-style combination), which yields an
    accurate mean and standard error without holding all samples at once.
    The local energy is hardcoded to the 2D Heisenberg model
    (:func:`local_energy_h2d`); use :func:`final_sampler_maker` for other
    Hamiltonians.

    Args:
        final_samples: Total number of evaluation samples (any remainder after
            even batching is sampled in a final smaller batch).
        batches: Number of independent sample batches.
        params: Trained model parameters.
        model: 2D wavefunction module exposing ``sample``.
        N: Linear lattice size (the lattice is N x N).

    Returns:
        Tuple (mean energy, standard error of the mean).
    """
    batch_size = final_samples // batches
    remainder = final_samples % batches
    mean = 0.0
    M2 = 0.0
    key1 = jax.random.key(1)

    for i in range(batches):
        print(f"Batch i={i+1} out of {batches}")
        key1, key2 = jax.random.split(key1)
        samples = model.apply(params, key1, batch_size, N, N, method="sample")
        log_amps = model.apply(params, samples)
        e_loc = jnp.real(local_energy_h2d(samples, params, model, log_amps))
        
        delta = jnp.mean(e_loc) - mean
        mean += delta / (i + 1)
        M2 += batch_size * jnp.var(e_loc) + delta**2 * batch_size * i / (i + 1)
        print(mean, '+/-', jnp.sqrt(M2) / (batch_size * (i + 1)))

    if remainder > 0:
        print(f"Leftovers :{remainder}")
        key1, key2 = jax.random.split(key1)
        samples = model.apply(params, key1, remainder, N, N, method="sample")
        log_amps = model.apply(params, samples)
        e_loc = jnp.real(local_energy_h2d(samples, params, model, log_amps))
        
        n1 = final_samples - remainder
        n2 = remainder
        n12 = final_samples
        delta = jnp.mean(e_loc) - mean
        mean += delta * n2 / n12
        M2 += n2 * jnp.var(e_loc) + delta**2 * n1 * n2 / n12

    e = (mean, jnp.sqrt(M2) / final_samples)
    return e


def local_energy_generator(hamiltonian: str, model: Any, params_ = None) -> Callable:
    """
    Returns a closed-over local energy function for the requested Hamiltonian.

    The returned callable has one of two signatures depending on ``params_``:

    - ``params_ is None`` (parameters stay dynamic, for training):
      ``local_energy_(samples, params, log_psi)``.
    - ``params_`` given (parameters frozen, for final evaluation):
      ``local_energy_(samples, log_psi)``.

    Args:
        hamiltonian: One of "h2d", "h2d_sym", "j1j2", "tfim", "cluster".
        model: Flax module evaluating the log-wavefunction.
        params_: Optional parameters to freeze into the closure.

    Returns:
        A local energy function matching one of the signatures above.

    Raises:
        ValueError: If ``hamiltonian`` is not a known key.
    """
    local_energy_func = {
        "h2d": local_energy_h2d,
        "h2d_sym": local_energy_h2d_sym,
        "j1j2": local_energy_j1j2,
        "tfim": local_energy_tfim,
        "cluster": local_energy_cluster
    }.get(hamiltonian)
    
    if local_energy_func is None:
        raise ValueError(f"Unknown Hamiltonian: {hamiltonian}")

    if params_ is None:
        def local_energy_(samples, params, log_psi):
            return local_energy_func(samples, params, model, log_psi)
    else:
        def local_energy_(samples, log_psi):
            return local_energy_func(samples, params_, model, log_psi)
    
    return local_energy_


def make_training_step(model: Any, optimizer: Any, local_energy_fn: Callable, get_jac: Callable,
                        numsamples: int, N: int, batches: int, samples_per_batch: int,
                        lambda_reg: float = 1e-1, is_2d: bool = True, is_complex: bool = False) -> Callable:
    """
    Creates a JIT-compiled minSR optimization step with Levenberg-Marquardt damping.

    Each call performs one full Variational Monte Carlo iteration:
    1. Sample ``numsamples`` configurations autoregressively and evaluate local energies.
    2. Compute the Jacobian of ln Psi over sample batches with ``jax.lax.map`` (bounded memory),
       flatten it to a dense matrix, center it column-wise, and scale by 1/sqrt(numsamples).
    3. Solve the damped normal equations (X X^T + lambda I) x = e_loc_c (or complex equivalent).
    4. Map the flat update back onto the parameter PyTree and apply it through ``optimizer``.

    Args:
        model: Wavefunction module exposing ``sample`` and ``__call__``.
        optimizer: Optax gradient transformation (e.g. SGD with momentum).
        local_energy_fn: Local energy closure with signature ``(samples, params, log_psi)``.
        get_jac: Callable computing Jacobians. For real models, returns a PyTree. For complex,
            returns a tuple (jac_r, jac_i).
        numsamples: Total Monte Carlo samples per step.
        N: Linear lattice size (or chain length for 1D).
        batches: Number of sequential Jacobian batches.
        samples_per_batch: Samples per batch (numsamples = batches * samples_per_batch).
        lambda_reg: Levenberg-Marquardt regularization damping parameter.
        is_2d: True for 2D square lattices (NxN), False for 1D chains (N).
        is_complex: True for complex wavefunctions (stacked real/imaginary Jacobians).

    Returns:
        A JIT-compiled step function ``(params, rng_key, opt_state) -> (params_new, opt_state_new, e_loc, rng_key_new)``.
    """
    def training_step(params: Any, rng_key: jax.random.PRNGKey, opt_state: Any) -> Tuple[Any, Any, jnp.ndarray, jax.random.PRNGKey]:
        rng_key, new_key = jax.random.split(rng_key)

        # Autoregressive sequence sampling + local energy evaluations
        if is_2d:
            samples = model.apply(params, new_key, numsamples, N, N, method="sample")
        else:
            samples = model.apply(params, new_key, numsamples, N, method="sample")

        log_probs = model.apply(params, samples)
        scale = 1.0 if is_complex else 0.5
        e_loc = local_energy_fn(samples, params, scale * log_probs)
        e_loc_c = e_loc - e_loc.mean()

        numsamples_ = samples.shape[0]
        samples_batched = samples.reshape(
            (batches, samples_per_batch, *samples.shape[1:])
        )

        if not is_complex:
            jacobian_batched = jax.lax.map(lambda s: get_jac(params, s), samples_batched)
            jacobian = jax.tree.map(
                lambda x: x.reshape(numsamples_, *x.shape[2:]), jacobian_batched
            )
            jac, tree, shapes, slices = _flatten_jacobian(jacobian, numsamples_)
            jac = jac - jnp.mean(jac, axis=0)
            jac = jac / jnp.sqrt(numsamples_)

            XdaggerX = jac @ jac.T
            Id = jnp.eye(XdaggerX.shape[0])
            x_solve = jax.scipy.linalg.solve(
                XdaggerX + lambda_reg * Id,
                e_loc_c,
                assume_a="pos",
            )
            tau = e_loc_c - lambda_reg * x_solve
            step_reg = 1.0 / jnp.maximum(jnp.linalg.norm(tau), 1e-12)
            dtheta = step_reg * jac.T @ x_solve
            grads = _unflatten_like_params(dtheta, tree, shapes, slices)
        else:
            # Complex model: get_jac returns (jac_r, jac_i)
            def batched_jac_fn(s):
                return get_jac(params, s)
            jac_r_b, jac_i_b = jax.lax.map(batched_jac_fn, samples_batched)
            jac_r_tree = jax.tree.map(lambda x: x.reshape(numsamples_, *x.shape[2:]), jac_r_b)
            jac_i_tree = jax.tree.map(lambda x: x.reshape(numsamples_, *x.shape[2:]), jac_i_b)

            jac_r, tree, shapes, slices = _flatten_jacobian(jac_r_tree, numsamples_)
            jac_r = jac_r - jnp.mean(jac_r, axis=0)
            jac_r = jac_r / jnp.sqrt(numsamples_)

            jac_i, _, _, _ = _flatten_jacobian(jac_i_tree, numsamples_)
            jac_i = jac_i - jnp.mean(jac_i, axis=0)
            jac_i = jac_i / jnp.sqrt(numsamples_)

            X = jnp.concatenate([jac_r, jac_i])
            ep = (e_loc - e_loc.mean()).conjugate() * (2.0 / jnp.sqrt(numsamples_))
            f = jnp.concatenate([jnp.real(ep), -1.0 * jnp.imag(ep)])

            XdaggerX = X @ X.T
            cfac = jax.scipy.linalg.cho_factor(XdaggerX + lambda_reg * jnp.eye(X.shape[0]))
            x_solve = jax.scipy.linalg.cho_solve(cfac, f)
            tau = f - lambda_reg * x_solve
            step_reg = 1.0 / jnp.linalg.norm(tau)
            dtheta = step_reg * X.T @ x_solve
            grads = _unflatten_like_params(dtheta, tree, shapes, slices)

        # Apply updates via standard optax optimizer
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)

        return params_new, opt_state_new, e_loc, new_key

    return jax.jit(training_step)


def make_training_step_adam(model: Any, optimizer: Any, local_energy_fn: Callable,
                            numsamples: int, N: int, is_2d: bool = True,
                            is_complex: bool = False) -> Callable:
    """
    Creates a JIT-compiled Adam optimization step for Variational Monte Carlo.

    Args:
        model: Wavefunction module.
        optimizer: Optax Adam optimizer.
        local_energy_fn: Local energy closure ``(samples, params, log_psi)``.
        numsamples: Total Monte Carlo samples per step.
        N: System size (N for 1D, NxN for 2D).
        is_2d: True for 2D square lattices, False for 1D chains.
        is_complex: True for complex wavefunctions.

    Returns:
        A JIT-compiled step function ``(params, rng_key, opt_state) -> (params_new, opt_state_new, e_loc, rng_key_new)``.
    """
    def get_loss(params, key):
        if is_2d:
            samples = model.apply(params, key, numsamples, N, N, method="sample")
        else:
            samples = model.apply(params, key, numsamples, N, method="sample")
        log_psi = model.apply(params, samples)
        scale = 1.0 if is_complex else 0.5
        e_loc = jax.lax.stop_gradient(local_energy_fn(samples, params, scale * log_psi))
        e_avg = e_loc.mean()
        if is_complex:
            loss = 2.0 * jnp.real(jnp.mean(jnp.conjugate(log_psi) * (e_loc - e_avg)))
        else:
            loss = jnp.mean(log_psi * e_loc - e_avg * log_psi)
        return loss, e_loc

    def training_step(params: Any, rng_key: jax.random.PRNGKey, opt_state: Any) -> Tuple[Any, Any, jnp.ndarray, jax.random.PRNGKey]:
        rng_key, new_key = jax.random.split(rng_key)
        (loss, e_loc), grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, e_loc, new_key

    return jax.jit(training_step)


def final_sampler_maker(model: Any, N: int, local_energy: Callable, params: Any,
                        is_2d: bool = True, is_complex: bool = False) -> Callable:
    """
    Returns a configured evaluation sampler that measures expectation values under static parameters.

    The sampler draws ``final_samples`` configurations in ``batches``
    independent chunks, evaluates the provided (frozen-parameter) local energy
    closure on each chunk, and merges the batch statistics online (Welford-style
    combination).

    Args:
        model: Wavefunction module exposing ``sample``.
        N: Linear lattice size (N for 1D, N x N for 2D).
        local_energy: Local energy closure with signature ``(samples, log_psi)``.
        params: Frozen model parameters captured in the returned closure.
        is_2d: True for 2D square lattices, False for 1D chains.
        is_complex: True for complex wavefunctions.

    Returns:
        A callable ``(final_samples, batches) -> (mean energy, standard error)``.
    """
    def final_sampler(final_samples: int, batches: int) -> Tuple[float, float]:
        key1 = jax.random.key(1)
        batch_size = final_samples // batches
        remainder = final_samples % batches
        mean = 0.0
        M2 = 0.0
        scale = 1.0 if is_complex else 0.5
        
        for i in range(batches): 
            key1, key2 = jax.random.split(key1)
            print(f"Batch i={i+1} out of {batches}")
            if is_2d:
                samples = model.apply(params, key1, batch_size, N, N, method="sample")
            else:
                samples = model.apply(params, key1, batch_size, N, method="sample")
            log_amps = model.apply(params, samples)
            e_loc = jnp.real(local_energy(samples, scale * log_amps))
            
            delta = jnp.mean(e_loc) - mean
            mean += delta / (i + 1)
            M2 += batch_size * jnp.var(e_loc) + delta**2 * batch_size * i / (i + 1)
            print(f"{float(mean):.6f} +/- {float(jnp.sqrt(M2) / (batch_size * (i + 1))):.3e}")
            
        if remainder > 0:
            print(f"Leftovers: {remainder}")
            key1, key2 = jax.random.split(key1)
            if is_2d:
                samples = model.apply(params, key1, remainder, N, N, method="sample")
            else:
                samples = model.apply(params, key1, remainder, N, method="sample")
            log_amps = model.apply(params, samples)
            e_loc = jnp.real(local_energy(samples, scale * log_amps))
            
            n1 = final_samples - remainder
            n2 = remainder
            n12 = final_samples
            delta = jnp.mean(e_loc) - mean
            mean += delta * n2 / n12
            M2 += n2 * jnp.var(e_loc) + delta**2 * n1 * n2 / n12
            
        e = (mean, jnp.sqrt(M2) / final_samples)
        return e
    return final_sampler
