"""
Recurrent Neural Network (RNN) Wavefunction Ansätze for Variational Monte Carlo (VMC).

This module implements 1D and 2D autoregressive neural quantum state (NQS) architectures
using JAX and Flax Linen:

1. :class:`RNNModel`:
   A 1D real-valued RNN wavefunction for stoquastic spin systems (e.g. 1D TFIM).
   Outputs normalized conditional log-amplitudes 0.5 * ln p(s).

2. :class:`CRNNModel`:
   A 1D complex-valued RNN wavefunction for non-stoquastic or topological spin chains
   (e.g. 1D cluster state). Jointly predicts log-probabilities and phase angles.

3. :class:`TwoDRNN`:
   A 2D recurrent cell receiving directional hidden state carries from left and upper
   spatial neighbors on a square lattice.

4. :class:`SequenceLayer`:
   A 2D sequence layer combining a :class:`TwoDRNN` cell with Gated Linear Unit (GLU)
   output transformations.

5. :class:`StackedPRNNModel`:
   A stacked 2D positive RNN wavefunction that samples configurations autoregressively
   along a zigzag path. Includes methods for optional C4v point-group symmetrization.

6. :class:`StackedCRNNModel`:
   A stacked 2D complex RNN wavefunction predicting both real log-amplitudes and
   complex phases for 2D frustrated spin models (e.g. J1-J2 Heisenberg).

Conventions:
    - Double precision (`float64`, `complex128`) enabled throughout.
    - Spin configurations are stored as integers s_i in {0, 1} and mapped to
      Pauli-z eigenvalues sigma_i = 2 s_i - 1.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn

# Force double precision for numerical stability in physical quantum simulations
jax.config.update("jax_enable_x64", True)
jax_dtype = jnp.float64


class CRNNModel(nn.Module):
    """
    1D Complex Recurrent Neural Network (CRNN) Wavefunction.

    Jointly parameterizes the log-amplitude and phase of a 1D spin chain:
        Psi(s) = sqrt(p(s)) * exp(i * phi(s))
        ln Psi(s) = 0.5 * ln p(s) + i * phi(s)

    The phase angle is bounded in [-pi, pi] using a soft-sign activation:
        phi(s) = pi * soft_sign(Dense(x)).

    Attributes:
        output_dim: Dimension of local Hilbert space (default: 2 for spin-1/2).
        num_hidden_units: Hidden recurrent state dimension.
        RNNcell_type: Recurrent cell type ('GRU', 'LSTM', or 'Vanilla').
    """

    output_dim: int = 2
    num_hidden_units: int = 20
    RNNcell_type: str = "GRU"

    def setup(self):
        """Initializes the recurrent cell and output projection layers."""
        if self.RNNcell_type == "GRU":
            self.cell = nn.GRUCell(
                name="gru_cell",
                features=self.num_hidden_units,
                kernel_init=jax.nn.initializers.glorot_uniform(),
                param_dtype=jax_dtype,
            )
        elif self.RNNcell_type == "LSTM":
            self.cell = nn.OptimizedLSTMCell(
                name="lstm_cell",
                features=self.num_hidden_units,
                kernel_init=jax.nn.initializers.glorot_uniform(),
                param_dtype=jax_dtype,
            )
        elif self.RNNcell_type == "Vanilla":
            self.cell = nn.SimpleCell(
                name="vanilla_cell",
                features=self.num_hidden_units,
                kernel_init=jax.nn.initializers.glorot_uniform(),
                param_dtype=jax_dtype,
            )
        else:
            raise ValueError(f"Invalid RNN cell type: {self.RNNcell_type}. Choose 'GRU', 'LSTM', or 'Vanilla'.")

        self.rnn = nn.RNN(self.cell, return_carry=True)
        self.dense = nn.Dense(
            self.output_dim,
            name="dense_layer",
            kernel_init=jax.nn.initializers.glorot_uniform(),
            param_dtype=jax_dtype,
        )
        self.dense_phase = nn.Dense(
            self.output_dim,
            name="dense_phase_layer",
            kernel_init=jax.nn.initializers.glorot_uniform(),
            param_dtype=jax_dtype,
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates the complex log-amplitudes ln Psi(s) for a batch of spin configurations.

        Args:
            inputs: Integer spin states of shape (numsamples, N) with values in {0, 1}.

        Returns:
            Complex log-amplitudes of shape (numsamples,), dtype complex128.
        """
        numsamples = inputs.shape[0]
        onehot_inputs = jax.nn.one_hot(inputs, num_classes=self.output_dim)

        # Autoregressive shift: input at site i is configuration up to site i-1
        shifted_onehot = jnp.roll(onehot_inputs, 1, axis=1)
        shifted_onehot = shifted_onehot.at[:, 0].set(jnp.zeros((numsamples, self.output_dim), dtype=jax_dtype))

        initial_carry = jnp.zeros((numsamples, self.num_hidden_units), dtype=jax_dtype)
        _, x = self.rnn(shifted_onehot, initial_carry=initial_carry)

        # Output projections
        logits = self.dense(x)
        phases = jnp.pi * nn.soft_sign(self.dense_phase(x))

        log_probs = nn.log_softmax(logits, axis=-1)
        total_log_prob = jnp.sum(log_probs * onehot_inputs, axis=(1, 2))
        total_phase = jnp.sum(phases * onehot_inputs, axis=(1, 2))

        return 0.5 * total_log_prob + 1j * total_phase

    def sample(self, key: jax.random.PRNGKey, numsamples: int, N: int) -> jnp.ndarray:
        """
        Draws exact configurations autoregressively from the wavefunction distribution.

        Args:
            key: JAX PRNG key.
            numsamples: Number of configurations to generate.
            N: Chain length (number of spins).

        Returns:
            Array of generated spin configurations of shape (numsamples, N).
        """
        inputs = jnp.zeros((numsamples, self.output_dim), dtype=jax_dtype)
        hidden_states = self.cell.initialize_carry(jax.random.key(1), inputs.shape)
        samples = jnp.zeros((numsamples, N), dtype=jax_dtype)
        keys = jax.random.split(key, N)

        for n in range(N):
            hidden_states, inputs = self.cell(hidden_states, inputs)
            logits = self.dense(inputs)
            sampled_site = jax.random.categorical(key=keys[n], logits=logits)
            samples = samples.at[:, n].set(sampled_site)
            inputs = jax.nn.one_hot(sampled_site, num_classes=self.output_dim)

        return samples


class TwoDRNN(nn.Module):
    """
    2D Recurrent Neural Network Cell.

    Combines incoming directional hidden carries from the left neighbor (horizontal)
    and upper neighbor (vertical) on a square lattice, applies a recurrent cell update,
    and projects the concatenated representation back to ``d_hidden`` via a linear map ``U``.

    Attributes:
        d_hidden: Hidden state dimension.
        d_model: Input and output feature dimensions.
        RNNcell_type: Recurrent cell architecture ('GRU', 'LSTM', or 'Vanilla').
    """

    d_hidden: int
    d_model: int
    RNNcell_type: str = "Vanilla"

    def setup(self):
        """Initializes directional cell and linear projection map."""
        if self.RNNcell_type == "GRU":
            self.cell = nn.GRUCell(
                name="gru_cell",
                features=self.d_hidden,
                kernel_init=jax.nn.initializers.glorot_uniform(),
                param_dtype=jax_dtype,
            )
        elif self.RNNcell_type == "LSTM":
            self.cell = nn.OptimizedLSTMCell(
                name="lstm_cell",
                features=self.d_hidden,
                kernel_init=jax.nn.initializers.glorot_uniform(),
                param_dtype=jax_dtype,
            )
        elif self.RNNcell_type == "Vanilla":
            self.cell = nn.SimpleCell(
                name="vanilla_cell",
                features=self.d_hidden,
                kernel_init=jax.nn.initializers.glorot_uniform(),
                param_dtype=jax_dtype,
            )
        else:
            raise ValueError(f"Invalid RNN cell type: {self.RNNcell_type}")

        self.U = self.param(
            "U",
            jax.nn.initializers.glorot_uniform(),
            (self.d_hidden * 2, self.d_hidden),
        )

    def __call__(self, inputs: Union[jnp.ndarray, Tuple[jnp.ndarray, ...]],
                 hidden_states: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward step combining horizontal and vertical neighbor information.

        Args:
            inputs: Incoming features from previous layer or lattice input.
            hidden_states: Tuple (h_left, h_up) of neighbor hidden carries.

        Returns:
            Tuple (new_hidden, new_hidden) providing output state and recurrent carry.
        """
        if isinstance(inputs, tuple):
            concatenated_inputs = jnp.concatenate(inputs, axis=-1)
        else:
            concatenated_inputs = inputs

        concatenated_hidden = jnp.concatenate(hidden_states, axis=-1)
        new_hidden_state, _ = self.cell(concatenated_hidden, concatenated_inputs)
        new_hidden_state = new_hidden_state @ self.U

        return new_hidden_state, new_hidden_state


class SequenceLayer(nn.Module):
    """
    Single 2D Recurrent Layer with Gated Linear Unit (GLU) Activation.

    Applies a spatial :class:`TwoDRNN` recurrent update followed by a GLU
    channel-wise gating mechanism:
        GLU(x) = Dense_1(x) * sigmoid(Dense_2(x))

    Attributes:
        RNN: The configured :class:`TwoDRNN` module.
        d_model: Dimensionality of the output feature representations.
    """

    RNN: TwoDRNN
    d_model: int

    def setup(self):
        """Initializes the recurrent block and GLU projection gates."""
        self.seq = self.RNN
        self.out1 = nn.Dense(self.d_model, param_dtype=jax_dtype)
        self.out2 = nn.Dense(self.d_model, param_dtype=jax_dtype)

    def __call__(self, inputs: Any, hidden_states: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Runs the spatial recurrent update followed by the GLU gating."""
        x, new_hidden_state = self.seq(inputs, hidden_states)
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
        return x, new_hidden_state


class StackedPRNNModel(nn.Module):
    """
    Stacked 2D Positive Recurrent Neural Network Wavefunction.

    Parameterizes positive ground-state wavefunctions on a 2D square lattice
    (e.g., Heisenberg antiferromagnet on bipartite lattices after Marshall sign transform):
        Psi(s) = sqrt(p(s)) > 0
        ln Psi(s) = 0.5 * ln p(s)

    Configurations are traversed sequentially along a serpentine (zigzag) path
    to minimize boundary cut distances across recurrent updates.

    Attributes:
        d_model: Feature dimension between sequence layers.
        d_hidden: Directional hidden state dimension in recurrent cells.
        n_layers: Number of stacked 2D sequence layers.
        RNNcell_type: Recurrent cell architecture ('GRU', 'LSTM', 'Vanilla').
    """

    d_model: int = 2
    d_hidden: int = 10
    n_layers: int = 1
    RNNcell_type: str = "Vanilla"

    def setup(self):
        """Builds stacked 2D sequence layers and dense spin classification head."""
        self.layers = [
            SequenceLayer(
                RNN=TwoDRNN(d_model=self.d_model, d_hidden=self.d_hidden, RNNcell_type=self.RNNcell_type),
                d_model=self.d_model,
            )
            for _ in range(self.n_layers)
        ]
        self.decoder = nn.Dense(2, param_dtype=jax_dtype)

    def generate_zigzag_path(self, Nx: int, Ny: int) -> List[Tuple[int, int]]:
        """Generates a serpentine path (x, y) visiting every site of an Nx x Ny lattice."""
        return [(i if j % 2 == 0 else Ny - 1 - i, j) for j in range(Ny) for i in range(Nx)]

    def __call__(self, samples: jnp.ndarray) -> jnp.ndarray:
        """
        Computes total log-probabilities ln p(s) = 2 * ln Psi(s) for a batch of configurations.

        Args:
            samples: Spin configurations of shape (numsamples, Nx, Ny) in {0, 1}.

        Returns:
            Log-probabilities of shape (numsamples,), dtype float64.
        """
        numsamples, Nx, Ny = samples.shape
        hidden_states = [
            [[jnp.zeros((numsamples, self.d_hidden), dtype=jax_dtype) for _ in range(Ny + 2)]
             for _ in range(Nx + 2)]
            for _ in range(self.n_layers)
        ]
        inputs = [
            [[jnp.zeros((numsamples, 2), dtype=jax_dtype) if k == 0
              else jnp.zeros((numsamples, self.d_model), dtype=jax_dtype)
              for _ in range(Ny + 2)]
             for _ in range(Nx + 2)]
            for k in range(self.n_layers + 1)
        ]

        samples_onehot = jnp.zeros((numsamples, Nx, Ny, 2), dtype=jax_dtype)
        cond_log_probs = jnp.zeros((numsamples, Nx, Ny, 2), dtype=jax_dtype)
        zigzag_path = self.generate_zigzag_path(Nx, Ny)

        for nx, ny in zigzag_path:
            for layer_index, layer in enumerate(self.layers):
                if layer_index == 0:
                    x1 = inputs[layer_index][nx - (-1)**ny][ny]
                    x2 = inputs[layer_index][nx][ny - 1]
                else:
                    x1 = inputs[layer_index][nx][ny]
                h1 = hidden_states[layer_index][nx - (-1)**ny][ny]
                h2 = hidden_states[layer_index][nx][ny - 1]
                inputs[layer_index + 1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1, x2), (h1, h2))

            x = self.decoder(inputs[-1][nx][ny])
            cond_log_probs = cond_log_probs.at[:, nx, ny].set(nn.log_softmax(x, axis=-1))
            inputs[0][nx][ny] = jax.nn.one_hot(samples[:, nx, ny], num_classes=2)
            samples_onehot = samples_onehot.at[:, nx, ny].set(inputs[0][nx][ny])

        return jnp.sum(cond_log_probs * samples_onehot, axis=(1, 2, 3))

    def logprobs_fromsymmetrygroup(self, list_samples: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Averages log-probabilities across elements of a spatial symmetry group via LogSumExp.

        Args:
            list_samples: List of rotated/reflected sample batches.

        Returns:
            Symmetrized log-probabilities of shape (numsamples,).
        """
        group_cardinal = len(list_samples)
        numsamples, Nx, Ny = list_samples[0].shape
        stacked_samples = jnp.reshape(jnp.concatenate(list_samples, axis=0), (-1, group_cardinal, Nx, Ny))

        def scan_c4v(carry, s):
            log_probs = self.__call__(s)
            return carry, log_probs

        scanned_func = nn.scan(
            scan_c4v,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        _, list_logprobs = scanned_func(0, stacked_samples)
        list_logprobs = jnp.reshape(list_logprobs, (group_cardinal, numsamples))

        return jax.scipy.special.logsumexp(list_logprobs, axis=0) - jnp.log(group_cardinal)

    def logprobs_c4vsym(self, samples: jnp.ndarray) -> jnp.ndarray:
        """
        Computes C4v symmetry-projected log-probabilities across the 8 dihedral transformations.

        Args:
            samples: Spin configurations of shape (numsamples, Nx, Ny).

        Returns:
            C4v-symmetrized log-probabilities of shape (numsamples,).
        """
        numsamples, Nx, Ny = samples.shape
        list_samples = [samples]
        list_samples.append(jnp.rot90(samples.reshape(-1, Nx, Ny, 1), k=-1, axes=(1, 2)).reshape(-1, Nx, Ny))
        list_samples.append(jnp.rot90(samples.reshape(-1, Nx, Ny, 1), k=-2, axes=(1, 2)).reshape(-1, Nx, Ny))
        list_samples.append(jnp.rot90(samples.reshape(-1, Nx, Ny, 1), k=-3, axes=(1, 2)).reshape(-1, Nx, Ny))
        list_samples.append(samples[:, ::-1])
        list_samples.append(samples[:, :, ::-1])
        list_samples.append(jnp.transpose(samples, axes=(0, 2, 1)))
        list_samples.append(jnp.transpose(list_samples[2], axes=(0, 2, 1)))

        return self.logprobs_fromsymmetrygroup(list_samples)

    def sample(self, key: jax.random.PRNGKey, numsamples: int, Nx: int, Ny: int) -> jnp.ndarray:
        """
        Autoregressively samples configurations site-by-site on an Nx x Ny lattice.

        Args:
            key: JAX PRNG key.
            numsamples: Number of configurations to generate.
            Nx, Ny: Lattice dimensions.

        Returns:
            Generated spin configurations of shape (numsamples, Nx, Ny).
        """
        samples = jnp.zeros((numsamples, Nx, Ny), dtype=jax_dtype)
        hidden_states = [
            [[jnp.zeros((numsamples, self.d_hidden), dtype=jax_dtype) for _ in range(Ny + 2)]
             for _ in range(Nx + 2)]
            for _ in range(self.n_layers)
        ]
        inputs = [
            [[jnp.zeros((numsamples, 2), dtype=jax_dtype) if k == 0
              else jnp.zeros((numsamples, self.d_model), dtype=jax_dtype)
              for _ in range(Ny + 2)]
             for _ in range(Nx + 2)]
            for k in range(self.n_layers + 1)
        ]

        zigzag_path = self.generate_zigzag_path(Nx, Ny)
        keys = jax.random.split(key, Nx * Ny)

        for idx, (nx, ny) in enumerate(zigzag_path):
            for layer_index, layer in enumerate(self.layers):
                if layer_index == 0:
                    x1 = inputs[layer_index][nx - (-1)**ny][ny]
                    x2 = inputs[layer_index][nx][ny - 1]
                else:
                    x1 = inputs[layer_index][nx][ny]
                h1 = hidden_states[layer_index][nx - (-1)**ny][ny]
                h2 = hidden_states[layer_index][nx][ny - 1]
                inputs[layer_index + 1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1, x2), (h1, h2))

            x = self.decoder(inputs[-1][nx][ny])
            sampled_site = jax.random.categorical(key=keys[idx], logits=nn.log_softmax(x, axis=-1))
            samples = samples.at[:, nx, ny].set(sampled_site)
            inputs[0][nx][ny] = jax.nn.one_hot(sampled_site, num_classes=2)

        return samples


class StackedCRNNModel(nn.Module):
    """
    Stacked 2D Complex Recurrent Neural Network (CRNN) Wavefunction.

    Parameterizes complex-valued wavefunctions on a 2D square lattice for
    frustrated quantum spin systems (e.g. 2D J1-J2 Heisenberg model):
        ln Psi(s) = 0.5 * ln p(s) + i * phi(s)

    Includes dual output decoder heads for log-probabilities and phase angles.

    Attributes:
        d_model: Feature dimension between sequence layers.
        d_hidden: Directional hidden state dimension in recurrent cells.
        n_layers: Number of stacked 2D sequence layers.
        RNNcell_type: Recurrent cell architecture ('GRU', 'LSTM', 'Vanilla').
    """

    d_model: int = 32
    d_hidden: int = 10
    n_layers: int = 1
    RNNcell_type: str = "Vanilla"

    def setup(self):
        """Initializes stacked 2D sequence layers, amplitude decoder, and phase decoder."""
        self.layers = [
            SequenceLayer(
                RNN=TwoDRNN(d_model=self.d_model, d_hidden=self.d_hidden, RNNcell_type=self.RNNcell_type),
                d_model=self.d_model,
            )
            for _ in range(self.n_layers)
        ]
        self.decoder = nn.Dense(2, param_dtype=jax_dtype)
        self.phase_decoder = nn.Dense(2, param_dtype=jax_dtype)

    def generate_zigzag_path(self, Nx: int, Ny: int) -> List[Tuple[int, int]]:
        """Generates a serpentine path (x, y) visiting every site of an Nx x Ny lattice."""
        return [(i if j % 2 == 0 else Ny - 1 - i, j) for j in range(Ny) for i in range(Nx)]

    def __call__(self, samples: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates complex log-amplitudes ln Psi(s) for a batch of configurations.

        Args:
            samples: Spin configurations of shape (numsamples, Nx, Ny) in {0, 1}.

        Returns:
            Complex log-amplitudes of shape (numsamples,), dtype complex128.
        """
        numsamples, Nx, Ny = samples.shape
        hidden_states = [
            [[jnp.zeros((numsamples, self.d_hidden), dtype=jax_dtype) for _ in range(Ny + 2)]
             for _ in range(Nx + 2)]
            for _ in range(self.n_layers)
        ]
        inputs = [
            [[jnp.zeros((numsamples, 2), dtype=jax_dtype) if k == 0
              else jnp.zeros((numsamples, self.d_model), dtype=jax_dtype)
              for _ in range(Ny + 2)]
             for _ in range(Nx + 2)]
            for k in range(self.n_layers + 1)
        ]

        samples_onehot = jnp.zeros((numsamples, Nx, Ny, 2), dtype=jax_dtype)
        cond_log_probs = jnp.zeros((numsamples, Nx, Ny, 2), dtype=jax_dtype)
        cond_phases = jnp.zeros((numsamples, Nx, Ny, 2), dtype=jax_dtype)
        zigzag_path = self.generate_zigzag_path(Nx, Ny)

        for nx, ny in zigzag_path:
            for layer_index, layer in enumerate(self.layers):
                if layer_index == 0:
                    x1 = inputs[layer_index][nx - (-1)**ny][ny]
                    x2 = inputs[layer_index][nx][ny - 1]
                else:
                    x1 = inputs[layer_index][nx][ny]
                h1 = hidden_states[layer_index][nx - (-1)**ny][ny]
                h2 = hidden_states[layer_index][nx][ny - 1]
                inputs[layer_index + 1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1, x2), (h1, h2))

            x = self.decoder(inputs[-1][nx][ny])
            phases = self.phase_decoder(x)
            cond_log_probs = cond_log_probs.at[:, nx, ny].set(nn.log_softmax(x, axis=-1))
            cond_phases = cond_phases.at[:, nx, ny].set(jnp.pi * nn.soft_sign(phases))
            inputs[0][nx][ny] = jax.nn.one_hot(samples[:, nx, ny], num_classes=2)
            samples_onehot = samples_onehot.at[:, nx, ny].set(inputs[0][nx][ny])

        log_probabilities = jnp.sum(cond_log_probs * samples_onehot, axis=(1, 2, 3))
        sum_phases = jnp.sum(cond_phases * samples_onehot, axis=(1, 2, 3))

        return 0.5 * log_probabilities + 1j * sum_phases

    def sample(self, key: jax.random.PRNGKey, numsamples: int, Nx: int, Ny: int) -> jnp.ndarray:
        """
        Autoregressively samples configurations site-by-site on an Nx x Ny lattice.

        Args:
            key: JAX PRNG key.
            numsamples: Number of configurations to generate.
            Nx, Ny: Lattice dimensions.

        Returns:
            Generated spin configurations of shape (numsamples, Nx, Ny).
        """
        samples = jnp.zeros((numsamples, Nx, Ny), dtype=jax_dtype)
        hidden_states = [
            [[jnp.zeros((numsamples, self.d_hidden), dtype=jax_dtype) for _ in range(Ny + 2)]
             for _ in range(Nx + 2)]
            for _ in range(self.n_layers)
        ]
        inputs = [
            [[jnp.zeros((numsamples, 2), dtype=jax_dtype) if k == 0
              else jnp.zeros((numsamples, self.d_model), dtype=jax_dtype)
              for _ in range(Ny + 2)]
             for _ in range(Nx + 2)]
            for k in range(self.n_layers + 1)
        ]

        zigzag_path = self.generate_zigzag_path(Nx, Ny)
        keys = jax.random.split(key, Nx * Ny)

        for idx, (nx, ny) in enumerate(zigzag_path):
            for layer_index, layer in enumerate(self.layers):
                if layer_index == 0:
                    x1 = inputs[layer_index][nx - (-1)**ny][ny]
                    x2 = inputs[layer_index][nx][ny - 1]
                else:
                    x1 = inputs[layer_index][nx][ny]
                h1 = hidden_states[layer_index][nx - (-1)**ny][ny]
                h2 = hidden_states[layer_index][nx][ny - 1]
                inputs[layer_index + 1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1, x2), (h1, h2))

            x = self.decoder(inputs[-1][nx][ny])
            sampled_site = jax.random.categorical(key=keys[idx], logits=nn.log_softmax(x, axis=-1))
            samples = samples.at[:, nx, ny].set(sampled_site)
            inputs[0][nx][ny] = jax.nn.one_hot(sampled_site, num_classes=2)

        return samples


class RNNModel(nn.Module):
    """
    1D Real-Valued Recurrent Neural Network (RNN) Wavefunction.

    Designed for stoquastic 1D spin Hamiltonians (e.g. Transverse-Field Ising Model):
        Psi(s) = sqrt(p(s)) > 0
        ln Psi(s) = 0.5 * ln p(s)

    Attributes:
        output_dim: Dimension of local Hilbert space (default: 2 for spin-1/2).
        num_hidden_units: Number of hidden units in the recurrent cell.
        RNNcell_type: Cell architecture ('GRU', 'LSTM', or 'Vanilla').
    """

    output_dim: int = 2
    num_hidden_units: int = 32
    RNNcell_type: str = "GRU"

    def setup(self):
        """Initializes the 1D recurrent cell and linear projection head."""
        if self.RNNcell_type == "GRU":
            self.cell = nn.GRUCell(
                name="gru_cell",
                features=self.num_hidden_units,
                kernel_init=jax.nn.initializers.glorot_uniform(),
                param_dtype=jax_dtype,
            )
        elif self.RNNcell_type == "LSTM":
            self.cell = nn.OptimizedLSTMCell(
                name="lstm_cell",
                features=self.num_hidden_units,
                kernel_init=jax.nn.initializers.glorot_uniform(),
                param_dtype=jax_dtype,
            )
        elif self.RNNcell_type == "Vanilla":
            self.cell = nn.SimpleCell(
                name="vanilla_cell",
                features=self.num_hidden_units,
                kernel_init=jax.nn.initializers.glorot_uniform(),
                param_dtype=jax_dtype,
            )
        else:
            raise ValueError(f"Invalid RNN cell type: {self.RNNcell_type}. Choose 'GRU', 'LSTM', or 'Vanilla'.")

        self.rnn = nn.RNN(self.cell, return_carry=True)
        self.dense = nn.Dense(
            self.output_dim,
            name="dense_layer",
            kernel_init=jax.nn.initializers.glorot_uniform(),
            param_dtype=jax_dtype,
        )

    def __call__(self, inputs: jnp.ndarray, initial_carry: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Computes total log-probabilities ln p(s) = 2 * ln Psi(s) for a batch of configurations.

        Args:
            inputs: Integer spin configurations of shape (numsamples, N) with values in {0, 1}.
            initial_carry: Optional initial carry state for the recurrent cell.

        Returns:
            Log-probabilities of shape (numsamples,), dtype float64.
        """
        numsamples = inputs.shape[0]
        onehot_inputs = jax.nn.one_hot(inputs, num_classes=self.output_dim)

        shifted_onehot = jnp.roll(onehot_inputs, 1, axis=1)
        shifted_onehot = shifted_onehot.at[:, 0].set(jnp.zeros((numsamples, self.output_dim), dtype=jax_dtype))

        carry, x = self.rnn(shifted_onehot, initial_carry=initial_carry)
        logits = self.dense(x)

        log_probs = nn.log_softmax(logits, axis=-1)
        return jnp.sum(log_probs * onehot_inputs, axis=(1, 2))

    def sample(self, key: jax.random.PRNGKey, numsamples: int, N: int) -> jnp.ndarray:
        """
        Autoregressively draws configurations site-by-site along the 1D spin chain.

        Args:
            key: JAX PRNG key.
            numsamples: Number of configurations to draw.
            N: Number of spins in the chain.

        Returns:
            Array of generated spin configurations of shape (numsamples, N).
        """
        inputs = jnp.zeros((numsamples, self.output_dim), dtype=jax_dtype)
        hidden_states = self.cell.initialize_carry(jax.random.key(1), inputs.shape)
        samples = jnp.zeros((numsamples, N), dtype=jax_dtype)
        keys = jax.random.split(key, N)

        for n in range(N):
            hidden_states, inputs = self.cell(hidden_states, inputs)
            logits = self.dense(inputs)
            sampled_site = jax.random.categorical(key=keys[n], logits=logits)
            samples = samples.at[:, n].set(sampled_site)
            inputs = jax.nn.one_hot(sampled_site, num_classes=self.output_dim)

        return samples