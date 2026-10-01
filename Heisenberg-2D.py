"""
2D Heisenberg Model on a Square Lattice: minSR and Adam Optimization.

This script trains a stacked 2D positive RNN wavefunction (:class:`StackedPRNNModel`)
on the 2D spin-1/2 Heisenberg model H = Sum_<i,j> S_i . S_j (open boundary
conditions) using Variational Monte Carlo (VMC) with either minimum-step
stochastic reconfiguration (minSR, natural gradient with Levenberg-Marquardt damping)
or the Adam optimizer.

Architecture and workflow:
    - Unified training step from ``Utils.utils.make_training_step`` / ``make_training_step_adam``,
    - Slurm-style wall-clock limit watchdog (``--time_limit``),
    - Full experiment bookkeeping: metadata, YAML config snapshot, model checkpoints,
      structured metric logging with ``Utils.utils.data_class`` (.npz), and diagnostic plots (.pdf),
    - Automatic initialization and caching under ``./init-params/``.

Usage:
    python Heisenberg-2D.py <key> [-t] [-grid] [--config PATH] [--time_limit D-HH:MM:SS]

where ``<key>`` seeds the run and names checkpoints, ``-t`` enables a quick smoke-test run,
and ``--config`` points to a YAML configuration file (default: ``Configs/config_h2d.yaml``).
"""

import argparse
from argparse import Namespace
import datetime
import math
import os
import pickle
import time
from typing import Any, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import yaml
import matplotlib.pyplot as plt

from Utils.models import StackedPRNNModel
from Utils.utils import (
    data_class,
    final_sampler_maker,
    local_energy_generator,
    make_training_step,
    make_training_step_adam,
    slurm_time_to_seconds,
)

# 1. Command-line argument parsing
parser = argparse.ArgumentParser(
    description="2D Heisenberg model VMC optimization using minSR or Adam."
)
parser.add_argument("key", type=int, help="Integer random seed and task identifier")
parser.add_argument("-t", "--test", action="store_true", help="Run a quick smoke test with tiny parameters")
parser.add_argument("-grid", action="store_true", help="Use Grid GRU variant (optional)")
parser.add_argument("--time_limit", type=str, default="2-10:00:00", help="Wall-clock time limit (e.g. D-HH:MM:SS)")
parser.add_argument("--config", type=str, default="Configs/config_h2d.yaml", help="Path to YAML configuration file")

args = parser.parse_args()

# 2. Load runtime YAML configuration
with open(args.config, "r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)

config = Namespace(**config_dict)

# 3. Parameter setup with smoke-test overrides
if args.test:
    N = 2
    dh = 2
    steps = 10
    batches = 2
    samples_per_batch = 3
    numsamples = batches * samples_per_batch
    numsamples_final = 10
    numsamples_check = 6
else:
    N = getattr(config, "N", 6)
    dh = getattr(config, "dh", 10)
    batches = getattr(config, "batches", 2)
    samples_per_batch = getattr(config, "samples_per_batch", getattr(config, "numsamples_per_batch", 50))
    numsamples = batches * samples_per_batch
    steps = getattr(config, "steps", 10000)
    numsamples_final = getattr(config, "numsamples_final", 20000)
    numsamples_check = getattr(config, "numsamples_check", 2000)

Nx = N
Ny = N
num_sites = N * N

lr = float(getattr(config, "lr", 6e-2))
m = float(getattr(config, "m", 0.2))
step_decay = int(getattr(config, "step_decay", 5000))
lambda_reg = float(getattr(config, "lambda_reg", 2e-5))
title = str(getattr(config, "title", "h2d-run"))
opt = str(getattr(config, "opt", getattr(config, "optimizer", "minsr"))).lower()

parsed_time = slurm_time_to_seconds(args.time_limit)
safety_margin = 6 * 3600 if (not args.test and parsed_time > 12 * 3600) else 0
TIME_LIMIT = max(parsed_time - safety_margin, 60)

T1 = time.time()

# 4. Directory and bookkeeping setup
experiment_tag = f"-test" if args.test else ""
folder = f"./Experiments/{opt.upper()}/2DH-{datetime.date.today()}-{title}{experiment_tag}"
print(f"Output folder: {folder}")

os.makedirs(folder, exist_ok=True)
os.makedirs(os.path.join(folder, "Data"), exist_ok=True)
os.makedirs(os.path.join(folder, "Plots"), exist_ok=True)
os.makedirs(os.path.join(folder, "Models"), exist_ok=True)

with open(os.path.join(folder, "config.yaml"), "w", encoding="utf-8") as f:
    yaml.dump(vars(config), f, default_flow_style=False, sort_keys=False)

run_docs = (
    f"2D Heisenberg Model VMC Optimization\n"
    f"Date: {datetime.date.today()}\n"
    f"N (lattice size): {N}x{N} ({num_sites} spins)\n"
    f"Hidden dimension (dh): {dh}\n"
    f"Optimizer: {opt}\n"
    f"Base learning rate: {lr}\n"
    f"Momentum: {m}\n"
    f"Step decay: {step_decay}\n"
    f"Lambda reg: {lambda_reg}\n"
    f"Samples per step: {numsamples} ({batches} batches x {samples_per_batch})\n"
    f"Total steps: {steps}\n"
    f"Time limit (sec): {TIME_LIMIT}\n"
)
with open(os.path.join(folder, "docs.txt"), "w", encoding="utf-8") as f:
    f.write(run_docs)

# 5. Initialize model and Hamiltonian operator
model = StackedPRNNModel(d_hidden=dh, d_model=2, n_layers=1, RNNcell_type="GRU")
local_energy = local_energy_generator("h2d", model)

key1, key2 = jax.random.split(jax.random.key(args.key))

os.makedirs("./init-params", exist_ok=True)
pkl_path = f"./init-params/h2d_params-{dh}.pkl"
try:
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)
except FileNotFoundError:
    dummy_input = jax.random.randint(key1, (2, Nx, Ny), 0, 2)
    params = model.init(key2, dummy_input)
    with open(pkl_path, "wb") as f:
        pickle.dump(params, f)

# 6. Define Jacobian and evaluation routines
def log_probs_fun(p: Any, s: jnp.ndarray) -> jnp.ndarray:
    """Computes real log-wavefunction amplitudes (0.5 * ln p(s))."""
    return 0.5 * model.apply(p, s)

def get_jac(p: Any, s: jnp.ndarray) -> Any:
    """Computes real Jacobian of log-amplitudes w.r.t. parameters."""
    return jax.jacrev(log_probs_fun)(p, s)

def eloc_final(p: Any, numsamples_f: int = 2000) -> Tuple[float, float]:
    """Evaluates checkpoint energy expectation and standard error."""
    key_f = jax.random.key(1)
    samples_f = model.apply(p, key_f, numsamples_f, Nx, Ny, method="sample")
    log_probs = model.apply(p, samples_f)
    e_loc = local_energy(samples_f, p, 0.5 * log_probs)
    e_mean = float(jnp.mean(e_loc))
    e_err = float(jnp.sqrt(jnp.var(e_loc)) / jnp.sqrt(numsamples_f))
    return e_mean, e_err

# 7. Optimizer schedule and training step configuration
def inverse_schedule(step_idx: int) -> float:
    """Inverse-time learning rate decay: max(lr / (1 + step/T), lr/100)."""
    return jnp.maximum(lr / (1.0 + step_idx / step_decay), lr / 100.0)

if opt == "minsr":
    optimizer = optax.sgd(learning_rate=inverse_schedule, momentum=m)
    train_step = make_training_step(
        model=model,
        optimizer=optimizer,
        local_energy_fn=local_energy,
        get_jac=get_jac,
        numsamples=numsamples,
        N=N,
        batches=batches,
        samples_per_batch=samples_per_batch,
        lambda_reg=lambda_reg,
        is_2d=True,
        is_complex=False,
    )
elif opt == "adam":
    optimizer = optax.adam(learning_rate=inverse_schedule)
    train_step = make_training_step_adam(
        model=model,
        optimizer=optimizer,
        local_energy_fn=local_energy,
        numsamples=numsamples,
        N=N,
        is_2d=True,
        is_complex=False,
    )
else:
    raise ValueError(f"Unsupported optimizer: '{opt}'. Use 'minsr' or 'adam'.")

# 8. Training loop
def train(params: Any, keynum: int = 0) -> Tuple[Any, data_class]:
    """
    Executes the training loop with periodic checkpoints, NaN watchdog, and wall-clock limit.
    """
    opt_state = optimizer.init(params)
    data = data_class(Energy=[], Var=[], Time=[])
    print(f"Training started ({opt.upper()}) for {steps} steps...")
    rng_key = jax.random.key(keynum)
    t0 = time.time()

    for i in range(steps):
        params, opt_state, e_loc, rng_key = train_step(params, rng_key, opt_state)
        e_mean = float(jnp.mean(e_loc))
        e_var = float(jnp.var(e_loc))
        elapsed = time.time() - t0
        data.update([e_mean, e_var, elapsed])

        if math.isnan(e_mean):
            print(f"Step {i:5d}: NaN detected in energy, terminating training.")
            break

        if time.time() - T1 > TIME_LIMIT:
            print(f"Step {i:5d}: Wall-clock time limit exceeded, terminating gracefully.")
            break

        # Periodic checkpoint
        if i % 5000 == 3 or (args.test and i == steps - 1):
            e_chk_mean, e_chk_err = eloc_final(params, numsamples_check)
            print(
                f"Step {i:5d}: Checkpoint saved | "
                f"E/site: {e_chk_mean/num_sites:.5f} +/- {e_chk_err/num_sites:.3e}"
            )
            metadata = {
                "opt": opt,
                "date": str(datetime.date.today()),
                "folder": folder,
                "dh": dh,
                "Ns-train": numsamples,
                "E/N": e_chk_mean / num_sites,
                "error/N": e_chk_err / num_sites,
                "Ns_check": numsamples_check,
            }
            chk_path = os.path.join(folder, f"Models/model_params-{opt}-s{i}-{args.key}.pkl")
            with open(chk_path, "wb") as f:
                pickle.dump(params, f)
            data.plot(
                "Var",
                folder=folder,
                opt=opt,
                i=args.key,
                log=True,
                label=f"{opt.upper()}, E/N: {e_chk_mean/num_sites:.5f} +/- {e_chk_err/num_sites:.3e}",
            )
            data.save(folder=folder, metadata=metadata, opt=opt, i=args.key)

    return params, data

# Run training
params, data = train(params, keynum=args.key)

# 9. Final high-precision evaluation and persistence
final_model_path = os.path.join(folder, f"Models/model_params-{steps}-{opt}-{args.key}.pkl")
with open(final_model_path, "wb") as f:
    pickle.dump(params, f)

local_energy_frozen = local_energy_generator("h2d", model, params)
final_sampler = final_sampler_maker(
    model=model,
    N=N,
    local_energy=local_energy_frozen,
    params=params,
    is_2d=True,
    is_complex=False,
)

batches_final = max((numsamples_final // 10000), 1)
e_loc_final_mean, e_loc_final_error = final_sampler(numsamples_final, batches_final)
e_mean_val = float(e_loc_final_mean)
e_err_val = float(e_loc_final_error)

metadata = {
    "opt": opt,
    "date": str(datetime.date.today()),
    "folder": folder,
    "N": N,
    "dh": dh,
    "E/N": e_mean_val / num_sites,
    "error/N": e_err_val / num_sites,
    "Ns_final": numsamples_final,
}
data.save(folder=folder, metadata=metadata, opt=opt, i=args.key)

data.plot(
    "Var",
    folder=folder,
    opt=opt,
    i=args.key,
    log=True,
    label=f"{opt.upper()}, E/N: {e_mean_val/num_sites:.5f} +/- {e_err_val/num_sites:.3e}",
)
data.plot(
    "Energy",
    folder=folder,
    opt=opt,
    i=args.key,
    log=False,
    label=f"{opt.upper()}, E/N: {e_mean_val/num_sites:.5f} +/- {e_err_val/num_sites:.3e}",
)

print(
    f"2D Heisenberg calculation completed successfully.\n"
    f"Final Ground State Energy / site: {e_mean_val/num_sites:.6f} +/- {e_err_val/num_sites:.3e}\n"
    f"Results stored under: {folder}"
)
