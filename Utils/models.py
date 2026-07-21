import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from typing import List, Tuple, Union, Optional, Callable, Any
from jax import jit
jax.config.update("jax_enable_x64", True)
jnp_dtype = jnp.float64
class CRNNModel(nn.Module):
    """
    RNN wavefunction
    """
    output_dim: int
    num_hidden_units: int
    RNNcell_type: str = "GRU"

    def setup(self):
      # Initialize the GRU cell with the specified number of hidden units
      if self.RNNcell_type == "GRU":
        self.cell = nn.GRUCell(
            name='gru_cell',
            features=self.num_hidden_units,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            #kernel_init = jax.nn.initializers.variance_scaling(1.0/self.num_hidden_units, "fan_avg", "uniform"),
            param_dtype = jnp.float64
        )
      elif self.RNNcell_type == "LSTM":
        self.cell = nn.OptimizedLSTMCell(
            name='lstm_cell',
            features=self.num_hidden_units,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            #kernel_init = jax.nn.initializers.variance_scaling(1.0/self.num_hidden_units, "fan_avg", "uniform"),
            param_dtype = jnp.float64
        )
      elif self.RNNcell_type == "Vanilla":
        self.cell = nn.SimpleCell(
            name='vanilla_cell',
            features=self.num_hidden_units,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            #kernel_init = jax.nn.initializers.variance_scaling(1.0/self.num_hidden_units, "fan_avg", "uniform"),
            #kernel_init = jax.nn.initializers.variance_scaling(1.0/self.model_scale, "fan_avg", "uniform"),
            param_dtype = jnp.float64
        )
      else:
        raise ValueError("Invalid RNN cell type")

      self.rnn = nn.RNN(self.cell, return_carry=True)
      self.dense = nn.Dense(
          self.output_dim,
          name = 'dense_layer',
          kernel_init = jax.nn.initializers.glorot_uniform(),
          param_dtype = jnp.float64
      )
      self.dense_phase = nn.Dense(
          self.output_dim,
          name = 'dense_phase_layer',
          kernel_init = jax.nn.initializers.glorot_uniform(),
          param_dtype = jnp.float64
      )

    def __call__(self, inputs):
        # Apply GRU layers
        onehot_inputs = jax.nn.one_hot(inputs, num_classes=self.output_dim)
        shifted_onehot_inputs = jnp.roll(onehot_inputs, 1, axis=1)
        shifted_onehot_inputs = shifted_onehot_inputs.at[:,0].set(jnp.zeros((inputs.shape[0],self.output_dim), dtype = jnp.float64))


        initial_carry = jnp.zeros((inputs.shape[0], self.num_hidden_units), dtype=jnp.float64)


        carry, x = self.rnn(shifted_onehot_inputs, initial_carry = initial_carry)

        # Output layer
        x = self.dense(x)
        # phases = self.dense_phase(x)
        phases = jnp.pi*nn.soft_sign(self.dense_phase(x))

        logits = nn.log_softmax(x, axis=-1)
        log_probabilities = jnp.sum(logits * onehot_inputs, axis = (1,2))
        sum_phases = jnp.sum(phases * onehot_inputs, axis = (1,2))
        return 0.5*log_probabilities + 1j*sum_phases

    def sample(self,key,numsamples,N):
        """Sample from the model for a given system size N and a number of samples `numsamples`"""
        inputs = jnp.zeros((numsamples,self.output_dim), dtype = jnp.float64)
        # hidden_states = jnp.zeros((numsamples,self.num_hidden_units), dtype = jnp.float64)
        hidden_states = self.cell.initialize_carry(jax.random.key(1), inputs.shape)

        samples_onehot = jnp.zeros((numsamples,N,self.output_dim), dtype = jnp.float64)
        samples = jnp.zeros((numsamples,N), dtype = jnp.float64)
        keys = jax.random.split(key, N) #pre-generate keys to get more randomness

        for n in range(N):
            hidden_states,inputs = self.cell(hidden_states,inputs)  # apply each layer
            inputs = self.dense(inputs)
            samples = samples.at[:,n].set(jax.random.categorical(key=keys[n], logits=inputs))
            inputs = jax.nn.one_hot(samples[:,n], num_classes=2)
        return samples

class TwoDRNN(nn.Module):
    """
    """
    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    RNNcell_type: str = "Vanilla"

    def setup(self):
      # Initialize the GRU cell with the specified number of hidden units
      if self.RNNcell_type == "GRU":
        self.cell = nn.GRUCell(
            name='gru_cell',
            features=self.d_hidden,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            param_dtype = jax_dtype
        )
      elif self.RNNcell_type == "  ":
        self.cell = nn.OptimizedLSTMCell(
            name='lstm_cell',
            features=self.d_hidden,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            param_dtype = jax_dtype
        )
      elif self.RNNcell_type == "Vanilla":
        self.cell = nn.SimpleCell(
            name='vanilla_cell',
            features=self.d_hidden,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            param_dtype = jax_dtype
        )
      else:
        raise ValueError("Invalid RNN cell type")

      self.U = self.param(
            "U",
            jax.nn.initializers.glorot_uniform(),
            (self.d_hidden*2, self.d_hidden),
        )

    def __call__(self, inputs, hidden_states):
        """Forward pass of a 2DRNN"""

        if isinstance(inputs, tuple):
          concatenate_inputs = jnp.concatenate(inputs, axis = -1)
        else:
          concatenate_inputs = inputs

        contatenated_hidden_states = jnp.concatenate(hidden_states, axis = -1)

        new_hidden_state,_ = self.cell(contatenated_hidden_states, concatenate_inputs)

        new_hidden_state = jax.vmap(lambda u: u @ self.U)(new_hidden_state)

        return new_hidden_state, new_hidden_state


class SequenceLayer(nn.Module):
    """Single RNN layer"""
    # Combining RNN for Softmax

    RNN: TwoDRNN  # 2dRNN module
    d_model: int  # model size

    def setup(self):
        """Initializes the RNN"""
        self.seq = self.RNN
        self.out1 = nn.Dense(self.d_model)
        self.out2 = nn.Dense(self.d_model)

    def __call__(self, inputs, hidden_states):
        x, new_hidden_state = self.seq(inputs, hidden_states)  # call LRU
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        # return inputs[0] + inputs[1] + x, new_hidden_state  # skip connection
        return x, new_hidden_state  # no skip connection

class StackedPRNNModel(nn.Module):
    """Encoder containing several SequenceLayer"""

    d_model: int
    d_hidden: int
    n_layers: int
    RNNcell_type: str = "Vanilla"

    def setup(self):
        self.layers = [
            SequenceLayer(
                RNN=TwoDRNN(d_model = self.d_model, d_hidden = self.d_hidden, RNNcell_type = self.RNNcell_type),
                d_model=self.d_model,
            )
            for _ in range(self.n_layers)
        ]
        self.decoder = nn.Dense(2)

    def generate_zigzag_path(self, Nx, Ny):
       return [(i if j % 2 == 0 else Ny - 1 - i, j) for j in range(Ny) for i in range(Nx)]

    def __call__(self, samples):
      """Sequential call of the model"""
      numsamples, Nx, Ny = samples.shape
      hidden_states = [[[jnp.zeros((numsamples,self.d_hidden), dtype = jax_dtype) for ny in range(-1,Ny+1)] for nx in range(-1,Nx+1)] for _ in range(self.n_layers)]
      inputs = [[[jnp.zeros((numsamples,2), dtype = jax_dtype) if k == 0 else jnp.zeros((numsamples,self.d_model), dtype = jax_dtype) for ny in range(-1, Ny+1) ] for nx in range(-1, Nx+1)] for k in range(self.n_layers+1)]
      samples_onehot = jnp.zeros((numsamples,Nx,Ny,2), dtype = jax_dtype)
      cond_log_probs = jnp.zeros((numsamples,Nx,Ny,2), dtype = jax_dtype)

      zigzag_path = self.generate_zigzag_path(Nx, Ny)

      for nx,ny in zigzag_path:
          for layer_index,layer in enumerate(self.layers):
              if layer_index == 0:
                x1 = inputs[layer_index][nx-(-1)**ny][ny]
                x2 = inputs[layer_index][nx][ny-1]
              else:
                x1 = inputs[layer_index][nx][ny]
              h1 = hidden_states[layer_index][nx-(-1)**ny][ny]
              h2 = hidden_states[layer_index][nx][ny-1]
              inputs[layer_index+1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1,x2), (h1, h2))  # apply each layer
          x = self.decoder(inputs[-1][nx][ny])
          cond_log_probs = cond_log_probs.at[:,nx,ny].set(nn.log_softmax(x, axis=-1))
          inputs[0][nx][ny] = jax.nn.one_hot(samples[:,nx,ny], num_classes=2)
          samples_onehot = samples_onehot.at[:,nx,ny].set(inputs[0][nx][ny])
      log_probabilities = jnp.sum(cond_log_probs * samples_onehot, axis = (1,2,3))
      return log_probabilities


    # def sample_and_logprobs(self,key,numsamples,Nx,Ny):
    #     """Sample from the model for a given system size Nx,Ny and a number of samples `numsamples` with log_probabilities output"""
    #     samples_onehot = jnp.zeros((numsamples,Nx, Ny, 2))
    #     samples = jnp.zeros((numsamples,Nx, Ny))
    #     hidden_states = [[[jnp.zeros((numsamples,self.d_hidden), dtype = jax_dtype) for ny in range(-1,Ny+1)] for nx in range(-1,Nx+1)] for _ in range(self.n_layers)]
    #     inputs = [[[jnp.zeros((numsamples,2), dtype = jax_dtype) if k == 0 else jnp.zeros((numsamples,self.d_model), dtype = jax_dtype) for ny in range(-1, Ny+1) ] for nx in range(-1, Nx+1)] for k in range(self.n_layers+1)]
    #     cond_log_probs = jnp.zeros((numsamples,Nx,Ny,2), dtype = jax_dtype)

    #     zigzag_path = self.generate_zigzag_path(Nx, Ny)

    #     keys = jax.random.split(key, Nx*Ny)

    #     for nx,ny in zigzag_path:
    #         for layer_index,layer in enumerate(self.layers):
    #             if layer_index == 0:
    #               x1 = inputs[layer_index][nx-(-1)**ny][ny]
    #               x2 = inputs[layer_index][nx][ny-1]
    #             else:
    #               x1 = inputs[layer_index][nx][ny]
    #             h1 = hidden_states[layer_index][nx-(-1)**ny][ny]
    #             h2 = hidden_states[layer_index][nx][ny-1]
    #             inputs[layer_index+1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1,x2), (h1, h2))  # apply each layer
    #         x = self.decoder(inputs[-1][nx][ny])
    #         cond_log_probs = cond_log_probs.at[:,nx,ny].set(nn.log_softmax(x, axis=-1))
    #         samples = samples.at[:,nx,ny].set(jax.random.categorical(key=keys[ny*Nx+nx], logits=cond_log_probs.at[:,nx,ny]))
    #         inputs[0][nx][ny] = jax.nn.one_hot(samples[:,nx,ny], num_classes=2)
    #         samples_onehot = samples_onehot.at[:,nx,ny].set(inputs[0][nx][ny])
    #     log_probabilities = jnp.sum(cond_log_probs * samples_onehot, axis = (1,2,3))

    #     return samples, log_probabilities

    # def logprobs_fromsymmetrygroup(self, list_samples):
    #     group_cardinal = len(list_samples)
    #     numsamples, Nx, Ny = list_samples[0].shape


    #     # Reshape and concatenate samples
    #     list_samples = jnp.reshape(jnp.concatenate(list_samples, axis=0), (-1, Nx, Ny))

    #     list_logprobs = self.__call__(list_samples)

    #     # Reshape and combine results
    #     list_logprobs = jnp.reshape(list_logprobs, (group_cardinal, numsamples))

    #     # Compute final log amplitude
    #     avg_logprobs = jax.scipy.special.logsumexp(list_logprobs, axis=0) - jnp.log(group_cardinal)

    #     return avg_logprobs
    def logprobs_fromsymmetrygroup(self, list_samples):
        group_cardinal = len(list_samples)
        numsamples, Nx, Ny = list_samples[0].shape
        
    
        # Reshape and concatenate samples
        list_samples = jnp.reshape(jnp.concatenate(list_samples, axis=0), (-1, group_cardinal, Nx, Ny))
 
        # list_logprobs = self.__call__(list_samples)

        def scan_c4v(self, carry, samples):
            log_probs = self.__call__(samples)
            return carry, log_probs
            
        scanned_func = nn.scan(
            scan_c4v,
            variable_broadcast='params', 
            split_rngs={'params': False}, 
            in_axes=0, 
            out_axes=0
        )
        
        _, list_logprobs = scanned_func(self, 0, list_samples)
        
        # Reshape and combine results
        list_logprobs = jnp.reshape(list_logprobs, (group_cardinal, numsamples))
            
        # Compute final log amplitude
        avg_logprobs = jax.scipy.special.logsumexp(list_logprobs, axis=0) - jnp.log(group_cardinal)
        
        return avg_logprobs
    def logprobs_c4vsym(self, samples):
        numsamples, Nx, Ny = samples.shape

        # # # Initialize list_samples with the original sample
        list_samples = [samples]


        # Apply rotations and reflections
        list_samples.append(jnp.rot90(samples.reshape(-1, Nx, Ny, 1), k=-1, axes=(1, 2)).reshape(-1, Nx, Ny))
        list_samples.append(jnp.rot90(samples.reshape(-1, Nx, Ny, 1), k=-2, axes=(1, 2)).reshape(-1, Nx, Ny))
        list_samples.append(jnp.rot90(samples.reshape(-1, Nx, Ny, 1), k=-3, axes=(1, 2)).reshape(-1, Nx, Ny))
        list_samples.append(samples[:, ::-1])  # Flip along rows
        list_samples.append(samples[:, :, ::-1])  # Flip along columns
        list_samples.append(jnp.transpose(samples, axes=(0, 2, 1)))  # Transpose samples
        list_samples.append(jnp.transpose(list_samples[2], axes=(0, 2, 1)))  # Transpose the 180-degree rotated samples

        # Call the method to compute the log problems with symmetry
        return self.logprobs_fromsymmetrygroup(list_samples)

    def sample(self,key,numsamples,Nx,Ny):
        """Sample from the model for a given system size Nx,Ny and a number of samples `numsamples`"""
        samples = jnp.zeros((numsamples,Nx, Ny))
        hidden_states = [[[jnp.zeros((numsamples,self.d_hidden), dtype = jax_dtype) for ny in range(-1,Ny+1)] for nx in range(-1,Nx+1)] for _ in range(self.n_layers)]
        inputs = [[[jnp.zeros((numsamples,2), dtype = jax_dtype) if k == 0 else jnp.zeros((numsamples,self.d_model), dtype = jax_dtype) for ny in range(-1, Ny+1) ] for nx in range(-1, Nx+1)] for k in range(self.n_layers+1)]

        zigzag_path = self.generate_zigzag_path(Nx, Ny)

        keys = jax.random.split(key, Nx*Ny)

        for nx,ny in zigzag_path:
            for layer_index,layer in enumerate(self.layers):
                if layer_index == 0:
                  x1 = inputs[layer_index][nx-(-1)**ny][ny]
                  x2 = inputs[layer_index][nx][ny-1]
                else:
                  x1 = inputs[layer_index][nx][ny]
                h1 = hidden_states[layer_index][nx-(-1)**ny][ny]
                h2 = hidden_states[layer_index][nx][ny-1]
                inputs[layer_index+1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1,x2), (h1, h2))  # apply each layer
            x = self.decoder(inputs[-1][nx][ny])
            samples = samples.at[:,nx,ny].set(jax.random.categorical(key=keys[ny*Nx+nx], logits=nn.log_softmax(x, axis=-1)))
            inputs[0][nx][ny] = jax.nn.one_hot(samples[:,nx,ny], num_classes=2)
        return samples

class StackedCRNNModel(nn.Module):
    """Encoder containing several SequenceLayer"""

    d_model: int
    d_hidden: int
    n_layers: int
    RNNcell_type: str = "Vanilla"

    def setup(self):
        self.layers = [
            SequenceLayer(
                RNN=TwoDRNN(d_model = self.d_model, d_hidden = self.d_hidden, RNNcell_type = self.RNNcell_type),
                d_model=self.d_model,
            )
            for _ in range(self.n_layers)
        ]
        self.decoder = nn.Dense(2)
        self.phase_decoder = nn.Dense(2)

    def generate_zigzag_path(self, Nx, Ny):
       return [(i if j % 2 == 0 else Ny - 1 - i, j) for j in range(Ny) for i in range(Nx)]

    def __call__(self, samples):
      """Sequential call of the model"""
      numsamples, Nx, Ny = samples.shape
      hidden_states = [[[jnp.zeros((numsamples,self.d_hidden), dtype = jax_dtype) for ny in range(-1,Ny+1)] for nx in range(-1,Nx+1)] for _ in range(self.n_layers)]
      inputs = [[[jnp.zeros((numsamples,2), dtype = jax_dtype) if k == 0 else jnp.zeros((numsamples,self.d_model), dtype = jax_dtype) for ny in range(-1, Ny+1) ] for nx in range(-1, Nx+1)] for k in range(self.n_layers+1)]
      samples_onehot = jnp.zeros((numsamples,Nx,Ny,2), dtype = jax_dtype)
      cond_log_probs = jnp.zeros((numsamples,Nx,Ny,2), dtype = jax_dtype)

      cond_phases = jnp.zeros((numsamples,Nx,Ny,2), dtype = jax_dtype)

      zigzag_path = self.generate_zigzag_path(Nx, Ny)

      for nx,ny in zigzag_path:
          for layer_index,layer in enumerate(self.layers):
              if layer_index == 0:
                x1 = inputs[layer_index][nx-(-1)**ny][ny]
                x2 = inputs[layer_index][nx][ny-1]
              else:
                x1 = inputs[layer_index][nx][ny]
              h1 = hidden_states[layer_index][nx-(-1)**ny][ny]
              h2 = hidden_states[layer_index][nx][ny-1]
              inputs[layer_index+1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1,x2), (h1, h2))  # apply each layer
          x = self.decoder(inputs[-1][nx][ny])
          phases = self.phase_decoder(x) #self.phase_decoder(inputs[-1][nx][ny])
          cond_log_probs = cond_log_probs.at[:,nx,ny].set(nn.log_softmax(x, axis=-1))
          # breakpoint()
          cond_phases = cond_phases.at[:,nx,ny].set(jnp.pi*nn.soft_sign(phases))
          inputs[0][nx][ny] = jax.nn.one_hot(samples[:,nx,ny], num_classes=2)
          samples_onehot = samples_onehot.at[:,nx,ny].set(inputs[0][nx][ny])
      log_probabilities = jnp.sum(cond_log_probs * samples_onehot, axis = (1,2,3))
      sum_phases = jnp.sum(cond_phases * samples_onehot, axis = (1,2,3))
      return log_probabilities * 0.5 + 1j * sum_phases

    def sample(self,key,numsamples,Nx,Ny):
        """Sample from the model for a given system size Nx,Ny and a number of samples `numsamples`"""
        samples = jnp.zeros((numsamples,Nx, Ny))
        hidden_states = [[[jnp.zeros((numsamples,self.d_hidden), dtype = jax_dtype) for ny in range(-1,Ny+1)] for nx in range(-1,Nx+1)] for _ in range(self.n_layers)]
        inputs = [[[jnp.zeros((numsamples,2), dtype = jax_dtype) if k == 0 else jnp.zeros((numsamples,self.d_model), dtype = jax_dtype) for ny in range(-1, Ny+1) ] for nx in range(-1, Nx+1)] for k in range(self.n_layers+1)]

        zigzag_path = self.generate_zigzag_path(Nx, Ny)

        keys = jax.random.split(key, Nx*Ny)

        for nx,ny in zigzag_path:
            for layer_index,layer in enumerate(self.layers):
                if layer_index == 0:
                  x1 = inputs[layer_index][nx-(-1)**ny][ny]
                  x2 = inputs[layer_index][nx][ny-1]
                else:
                  x1 = inputs[layer_index][nx][ny]
                h1 = hidden_states[layer_index][nx-(-1)**ny][ny]
                h2 = hidden_states[layer_index][nx][ny-1]
                inputs[layer_index+1][nx][ny], hidden_states[layer_index][nx][ny] = layer((x1,x2), (h1, h2))  # apply each layer
            x = self.decoder(inputs[-1][nx][ny])
            samples = samples.at[:,nx,ny].set(jax.random.categorical(key=keys[ny*Nx+nx], logits=nn.log_softmax(x, axis=-1)))
            inputs[0][nx][ny] = jax.nn.one_hot(samples[:,nx,ny], num_classes=2)

        return samples

class RNNModel(nn.Module): #no shape inference
    """
    RNN wavefunction
    """
    output_dim: int
    num_hidden_units: int
    RNNcell_type: str = "GRU"



    def setup(self):
      # Initialize the GRU cell with the specified number of hidden units
      if self.RNNcell_type == "GRU":
        self.cell = nn.GRUCell(
            name='gru_cell',
            features=self.num_hidden_units,
            kernel_init = jax.nn.initializers.glorot_uniform(), #what is this? It initializes weights optimally
            param_dtype = jnp_dtype
        )
      elif self.RNNcell_type == "LSTM":
        self.cell = nn.OptimizedLSTMCell(
            name='lstm_cell',
            features=self.num_hidden_units,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            param_dtype = jnp_dtype
        )
      elif self.RNNcell_type == "Vanilla":
        self.cell = nn.SimpleCell(
            name='vanilla_cell',
            features=self.num_hidden_units,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            param_dtype = jnp_dtype
        )
      else:
        raise ValueError("Invalid RNN cell type")

      self.rnn = nn.RNN(self.cell, return_carry=True)
      self.dense = nn.Dense( #why a dense layer? outp
          self.output_dim,
          name = 'dense_layer',
          kernel_init = jax.nn.initializers.glorot_uniform(),
          param_dtype = jnp_dtype
      )

    def __call__(self, inputs, initial_carry=None):
        # Apply GRU layers
        onehot_inputs = jax.nn.one_hot(inputs, num_classes=self.output_dim)
        shifted_onehot_inputs = jnp.roll(onehot_inputs, 1, axis=1) #shift the input to validate
        shifted_onehot_inputs = shifted_onehot_inputs.at[:,0].set(jnp.zeros((inputs.shape[0],self.output_dim), dtype = jnp_dtype))

        carry, x = self.rnn(shifted_onehot_inputs, initial_carry = initial_carry)

        # Output layer
        x = self.dense(x)

        logits = nn.log_softmax(x, axis=-1)
        # logits = SMTaylor(x, m=0.8)
        log_probabilities = jnp.sum(logits * onehot_inputs, axis = (1,2))
        return log_probabilities

    def sample(self,key,numsamples,N):
        """Sample from the model for a given system size N and a number of samples `numsamples`"""
        inputs = jnp.zeros((numsamples,self.output_dim), dtype = jnp_dtype)
        # hidden_states = jnp.zeros((numsamples,self.num_hidden_units), dtype = jnp.float32)
        hidden_states = self.cell.initialize_carry(jax.random.key(1), inputs.shape)

        samples_onehot = jnp.zeros((numsamples,N,self.output_dim), dtype = jnp_dtype)
        samples = jnp.zeros((numsamples,N), dtype = jnp_dtype)
        keys = jax.random.split(key, N) #pre-generate keys to get more randomness

        for n in range(N):
            hidden_states,inputs = self.cell(hidden_states,inputs)  # apply each layer
            inputs = self.dense(inputs)
            # samples = samples.at[:,n].set(jax.random.categorical(key=keys[n], logits=SMTaylor(inputs, m=0.8)))
            samples = samples.at[:,n].set(jax.random.categorical(key=keys[n], logits=inputs))
            inputs = jax.nn.one_hot(samples[:,n], num_classes=2)
        return samples
