
# @title Imports
import math
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Tuple, Union, Optional, Callable, Any
import optax
from tqdm import tqdm
from functools import partial
from jax import jit
import time
from math import ceil
import matplotlib.pyplot as plt
jax_dtype = jnp.float64
# import optuna
import json
import matplotlib.pyplot as plt
import datetime, os, sys
import pickle
nss = [20,80,120,160,240,410,490,620,830,1240]
# nss = [80,160,410]
import argparse
import csv
# 1. Initialize the parser
parser = argparse.ArgumentParser(description="A sample command-line parser.")
# lrs = [0.08,0.04,0.02,0.005,0.003,0.001]
csv_file = f"CSV/abl-j1j2-nss.csv"
# 2. Add expected arguments
parser.add_argument("key", type=int, help="Your name (positional argument)")
parser.add_argument("-t", "--test", action="store_true", help="Increase output verbosity")
args = parser.parse_args()
# bools_tr = [True, False]

if args.test:
    test = True
else:
    test = False


opt_dict = {'opt': 'minsr',
                'trust_region': True,
                 'CN-1': False,
                 'CN-2': False,
                 'lr': 6e-2,
                 'm': 0.2,
                 'step_decay': 5000,
                 'min_lr': 1e-5,
                 'TIME_LIMIT': 1*3600
                 }
if args.test:
   dh = 1
     #  lrs=[0.05,0.01]
   nss=[6,8]
   batches = 2 #batching jacobian computation to save memory.
   samples_per_batch = nss[args.key]//2
   numsamples = batches * samples_per_batch
   steps = 2
   N = 2
   iter_num = 2
  #  iter_list = [3,4,5]
  #  N_list = [2,3,2]
   numsamples_f = 2
else:
  dh = 10
  batches = 2 #batching jacobian computation to save memory.
  samples_per_batch = nss[args.key]//2
  numsamples = batches * samples_per_batch
  steps = int(2e6)
  N=6
  iter_num= 2
  # iter_list = [1000,2000,2000,1000,10000]
  # N_list = [6,8,10,12,10]
  numsamples_f = int(2e4)

TIME_LIMIT = opt_dict['TIME_LIMIT']
k = int(sys.argv[1])
# lrs= [0.03,0.0,0.05,0.06]
lr = opt_dict['lr']
m = opt_dict['m']
bool_tr = int(opt_dict['trust_region'])
step_decay=opt_dict['step_decay']
# lr = 5e-4
J1 = 1
J2 = 0.5

opt = 'minsr'
title = f'-minsr-ls-IR'
s = f'''Model: J1/J2 model. 
Number of samples: {numsamples}.
hidden vector dimension: {dh}.
steps: {steps}
regularization: step CN-1 0.25 plus e
lr: {lr}
m: {m}
adam_lr: 2.61e-2
J2/J1: {J2}F
N: {N}
'''
# def matrix_init(key, shape, dtype=jax_dtype, normalization=1):
#     return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization



# folder = './Experiments/J1J2/J1J2-' + str(datetime.date.today()) + title

# if not os.path.exists(folder):
#     os.mkdir(folder)
#     os.mkdir(folder+'/Data')
#     os.mkdir(folder+'/Plots')
#     os.mkdir(folder+'/Models')


# with open(folder + '/docs.txt','w') as f:
#     f.write(s)


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
            param_dtype = jnp.float64
        )
      elif self.RNNcell_type == "LSTM":
        self.cell = nn.OptimizedLSTMCell(
            name='lstm_cell',
            features=self.d_hidden,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            param_dtype = jnp.float64
        )
      elif self.RNNcell_type == "Vanilla":
        self.cell = nn.SimpleCell(
            name='vanilla_cell',
            features=self.d_hidden,
            kernel_init = jax.nn.initializers.glorot_uniform(),
            param_dtype = jnp.float64
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
        # self.phase1 = nn.Dense(self.d_model)#added phases
        # self.phase2 = nn.Dense(self.d_model)#added phases

    def __call__(self, inputs, hidden_states):
        x, new_hidden_state = self.seq(inputs, hidden_states)  # call LRU
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        #add phases

        # return inputs[0] + inputs[1] + x, new_hidden_state  # skip connection
        return x, new_hidden_state  # no skip connection

class StackedRNNModel(nn.Module):
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

# @title local energy
def local_energy(samples, params, model, log_psi) -> List[float]:

    """Computes the local energy of the 2D Heisenberg model"""

    numsamples,Nx,Ny = samples.shape

    N = Nx*Ny

    local_energies = jnp.zeros((numsamples), dtype = jax_dtype)


    # for i in range(Nx-1): #diagonal elements (right neighbours)

    #     spins_products = 0.25*(2*samples[:,i]-1)*(2*samples[:,i+1]-1)
    #     local_energies += jnp.sum(jnp.copy(spins_products), axis = 1)

    # for j in range(Ny-1): #diagonal elements (upward neighbours (or downward, it depends on the way you see the lattice))

    #     spins_products = 0.25*(2*samples[:,:,j]-1)*(2*samples[:,:,j+1]-1)
    #     local_energies += jnp.sum(jnp.copy(spins_products), axis = 1)

    # # for j in range(Ny-1): #diagonal elements left diag attempt

    # #     spins_products = 0.25*(2*samples[:,:,j]-1)*(2*jnp.roll(samples[:,:,j+1],1,axis=1)-1)
    # #     local_energies += J2 *jnp.sum(jnp.copy(spins_products), axis = 1)

    # # for j in range(1,Ny): #diagonal elements right diag attempt

    # #     spins_products = 0.25*(2*samples[:,:,j]-1)*(2*jnp.roll(samples[:,:,j-1],-1,axis=1)-1)
    # #     local_energies += J2 * jnp.sum(jnp.copy(spins_products), axis = 1)
    # e_loc = []
    # for j in range(numsamples): #diagonal elements right and left at the same time for second nearest neighbor naive
    #     spins = 2*samples[j]-1

    #     e=0
    #     for i in range(Ny-1):
    #         e += 0.25*jnp.sum(spins[i]* jnp.roll(spins[i+1],1) + spins[i]* jnp.roll(spins[i+1],-1))
    #         e -= 0.52*spins[i][0]*spins[i+1][-1] + spins[i][-1]*spins[i+1][0] # removing boundary terms for obc
    #     e_loc.append(e)
    # local_energies+= J2*jnp.array(e_loc)
    sigmap = 2 * samples - 1
    local_energies+=0.25*J1*jnp.sum(sigmap[:,:,:-1]*sigmap[:,:,1:],axis=(1,2)) #right
    local_energies+=0.25*J1*jnp.sum(sigmap[:,:-1,:]*sigmap[:,1:,:],axis=(1,2)) #down
    local_energies+=0.25*J2*jnp.sum(sigmap[:,:-1,:-1]*sigmap[:,1:,1:],axis=(1,2)) #right diagonal
    local_energies+=0.25*J2*jnp.sum(sigmap[:,:-1,1:]*sigmap[:,1:,:-1],axis=(1,2)) #left diagonal

    def step_fn_horizontal(n, state):

        s, output = state
        _, Nx,Ny = s.shape

        i = (n//Ny) #set back to zero when equal to Nx-1
        j = n%Ny

        flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
        flipped_state = flipped_state.at[:, i+1,j].set(1 - flipped_state[:, i+1,j])
        flipped_logpsi = model.apply(params,flipped_state)
        output += (s[:, i,j] + s[:, i+1,j] == 1) *(-0.5)* jnp.exp(flipped_logpsi - log_psi)

        return s, output

    def step_fn_right(n, state):

        s, output = state
        _, Nx,Ny = s.shape

        i = (n//(Ny-1)) #set back to zero when equal to Nx-1
        j = n%(Ny-1)

        flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
        flipped_state = flipped_state.at[:, i+1,j+1].set(1 - flipped_state[:, i+1,j+1])
        flipped_logpsi = model.apply(params,flipped_state) #No 1/2 here
        output += J2 * (s[:, i,j] + s[:, i+1,j+1] == 1) *(0.5)* jnp.exp(flipped_logpsi - log_psi)

        return s, output
    def step_fn_left(n, state):

      s, output = state
      _, Nx,Ny = s.shape

      i = (n//(Ny-1)) #set back to zero when equal to Nx-1
      j = n%(Ny-1)
      j+=1

      flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
      flipped_state = flipped_state.at[:, i+1,j-1].set(1 - flipped_state[:, i+1,j-1])
      flipped_logpsi = model.apply(params,flipped_state)
      output += J2 * (s[:, i,j] + s[:, i+1,j-1] == 1) *(0.5)* jnp.exp(flipped_logpsi - log_psi)

      return s, output

    def step_fn_vertical(n, state):

        s, output = state
        _, Nx,Ny = s.shape

        j = (n//Nx) #set back to zero when equal to Nx-1
        i = n%Nx


        flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
        flipped_state = flipped_state.at[:, i,j+1].set(1 - flipped_state[:, i,j+1])
        flipped_logpsi = model.apply(params,flipped_state)
        output += ((s[:, i,j] + s[:, i,j+1] == 1)*(-0.5))*jnp.exp(flipped_logpsi - log_psi)

        return s, output

    # Off Diagonal Term
    output = jnp.zeros((numsamples), dtype=jnp.complex128)
    _, off_diag_term_vertical = jax.lax.fori_loop(0, Nx*(Ny-1), step_fn_vertical, (samples, output))
    _, off_diag_term_horizontal = jax.lax.fori_loop(0, (Nx-1)*(Ny), step_fn_horizontal, (samples, output))
    _, off_diag_term_right = jax.lax.fori_loop(0, (Nx-1)*(Ny-1), step_fn_right, (samples, output))
    _, off_diag_term_left = jax.lax.fori_loop(0, (Ny-1)*(Nx-1), step_fn_left, (samples, output))

    local_energies += off_diag_term_vertical +  off_diag_term_horizontal + off_diag_term_left + off_diag_term_right

    return local_energies

def energy_estimate(params, key, numsamples, Nx, Ny, model):
    final_samples = model.apply(params,rng_key,numsamples,Nx, Ny,method="sample")
    log_probs = model.apply(params,final_samples)#, method="logprobs_c4vsym")
    e_loc_final = local_energy(final_samples, params, model, 0.5*log_probs)
    e_loc_final_mean = jnp.mean(e_loc_final)
    e_loc_final_error = jnp.var(e_loc_final)/jnp.sqrt(numsamples)
    return e_loc_final_mean, e_loc_final_error

model = StackedRNNModel(d_hidden= dh, d_model=32, n_layers=1, RNNcell_type = "GRU")

# @title
def log_probs_fun_r(params, samples):
    return jnp.real(model.apply(params,samples))
def log_probs_fun_i(params, samples):
    return jnp.imag(model.apply(params,samples))
#param_count = sum(x.size for x in jax.tree_leaves(params))

def get_jac_i(params, samples):
  jacobian = jax.jacrev(log_probs_fun_i)(params, samples)
  return jacobian

def get_jac_r(params, samples):
  jacobian = jax.jacrev(log_probs_fun_r)(params, samples)
  return jacobian

def get_minSR_gradients(params, samples, local_energies,CN):
  # jacobian_r = jax.jacrev(log_probs_fun_r)(params, samples)
  # jacobian_i = jax.jacrev(log_probs_fun_i)(params, samples)
  p = batches
  numsamples = samples.shape[0]
  samples_batched = samples.reshape((batches, samples_per_batch, samples.shape[1], samples.shape[2]))
  # jax.lax.map is used instead of vmap to avoid memory issues, as vmap would try to compute the entire jacobian at once
  jacobian_batched_i = jax.lax.map(lambda samples: get_jac_i(params, samples), samples_batched)
  jacobian_batched_r = jax.lax.map(lambda samples: get_jac_r(params, samples), samples_batched)
  # Reshape the batched jacobian back to the original shape
  jacobian_i = jax.tree.map(lambda x: x.reshape(numsamples, *x.shape[2:]), jacobian_batched_i)
  jacobian_r = jax.tree.map(lambda x: x.reshape(numsamples, *x.shape[2:]), jacobian_batched_r)

  numsamples = samples.shape[0]

  flattened_jac_r, tree = jax.tree_util.tree_flatten(jacobian_r)
  flattened_jac_i, tree = jax.tree_util.tree_flatten(jacobian_i)

  shapes = [it.shape for it in flattened_jac_r]

  slices = []
  last = flattened_jac_r[0][0].size
  slices.append(slice(0,last))
  for it in flattened_jac_r[1:]:
      slices.append(slice(last,last+it[0].size))
      last += it[0].size

  jac_r = jnp.concatenate([it.reshape(it.shape[0],-1) for it in flattened_jac_r], axis=-1)
  jac_r -= jnp.mean(jac_r, axis = 0)
  jac_r = jac_r/ jnp.sqrt(numsamples)

  shapes = [it.shape for it in flattened_jac_i]



  slices = []
  last = flattened_jac_i[0][0].size
  slices.append(slice(0,last))
  for it in flattened_jac_i[1:]:
      slices.append(slice(last,last+it[0].size))
      last += it[0].size

  jac_i = jnp.concatenate([it.reshape(it.shape[0],-1) for it in flattened_jac_i], axis=-1)
  jac_i -= jnp.mean(jac_i, axis = 0)
  jac_i = jac_i/ jnp.sqrt(numsamples)
  X = jnp.concatenate([jac_r,jac_i])

  ep = (local_energies - local_energies.mean()).conjugate()* ( 2 / jnp.sqrt(numsamples))
  e_r = jnp.real(ep)
  e_i = jnp.imag(ep)
  f = jnp.concatenate([e_r,-1*e_i])

  # S_matrix = jac.T @ jac #XXdagger #you can comment or uncomment

  # lambda_reg = jnp.var(jac)
  # lambda_reg = jnp.trace(XdaggerX)
  norm  = jnp.linalg.norm(X@X.T)
  lambda_reg = 2e-5
  # print(lambda_reg)
#   XdaggerX_inv =  jax.scipy.linalg.inv((X@X.T + lambda_reg * jnp.eye(X.shape[0])))
  # XdaggerX_inv = minSR_pseudo_inverse(XdaggerX, soft=True)
#   norm_inv  = jnp.linalg.norm(XdaggerX_inv)
  rank = jnp.linalg.matrix_rank(X@X.T)
  CN_t = jnp.log(norm)

  cfac = jax.scipy.linalg.cho_factor(X@X.T + lambda_reg * jnp.eye(X.shape[0]))
  step_reg =(10.5/jnp.linalg.norm(local_energies - local_energies.mean()))**(bool_tr)*1/(1+abs(CN_t-CN)**0.25)#1/(1+abs(CN_t-CN)**0.25)#*1/(1+abs(CN_t-CN)**0.25)#2**0.5/( CN/CN_t + CN_t/CN)**0.5*0.5/( CN/CN_t + CN_t/CN)**0.5 * 
  gradients =  step_reg *  X.T @ jax.scipy.linalg.cho_solve(cfac, f)
  #grad_norm = jnp.linalg.norm(gradients)

  #gradients *= jnp.min(1,5/grad_norm)
  CN = CN_t

  ### unflatten
  flat_tree = []
  for shape, _slice in zip(shapes, slices):
      flat_tree.append(gradients[_slice].reshape(shape[1:]))

  original_grad = jax.tree_util.tree_unflatten(tree, flat_tree)

  # stats = {
  #     "CN":CN,
  #     "grad":jnp.linalg.norm(gradients),
  #     "norm_inv":norm_inv
  # }

  return original_grad, CN, rank
  # return original_grad, S_matrix

def get_loss(params, key, numsamples, Nx,Ny, model):

  samples = model.apply(params,key,numsamples,Nx,Ny, method="sample")
  log_amps = model.apply(params,samples)

  e_loc = jax.lax.stop_gradient(local_energy(samples, params, model, log_amps))
  e_avg = e_loc.mean()

  # loss = jnp.mean(jnp.multiply(log_probs, e_loc) - jnp.multiply(log_probs, e_avg))
  loss = 2*jnp.real(jnp.mean(jnp.conjugate(log_amps)*(e_loc-e_avg)))
  return loss, e_loc

def eloc_final(params,numsamples_f = 3*10**4):
    key_f = jax.random.key(1)
    # numsamples_f = 10**4
    samples_f = model.apply(params,key_f,numsamples_f,Nx, Ny,method="sample")
    log_probs = model.apply(params,samples_f)#,method="logprobs_c4vsym")
    e_loc = local_energy(samples_f, params, model, log_probs)
    e_loc_mean = jnp.mean(e_loc)
    e_loc_err =jnp.sqrt( jnp.var(jnp.abs(e_loc)))/ jnp.sqrt(numsamples_f)
    E = (jnp.real(e_loc_mean), jnp.real(e_loc_err))
    return E

@partial(jit, static_argnums=(3,))
def step_adam(params, rng_key, opt_state, get_loss=get_loss):
    rng_key, new_key = jax.random.split(rng_key)
    value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, numsamples, Nx, Ny, model)
    # grads_flat, _ = jax.tree_util.tree_flatten(grads)
    # global_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
    # g_factor = jnp.minimum(1.0, grad_clip_norm / global_l2)
    # grads = jax.tree_util.tree_map(lambda g: g * g_factor, grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value, new_key

def train_minsr(params, steps, keynum=0, energies =[], vars = [], times = [], CNs = [], ranks = []):
    @partial(jit)
    def step(params, rng_key, opt_state, CN):
        grads, e_loc, CN, rank = get_grad(params, rng_key, numsamples, Nx, Ny, model, CN)
        # grads, e_loc, S_matrix = get_grad(params, new_key, numsamples, Nx, Ny, model)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, e_loc, CN, rank
    # return params, opt_state, e_loc, S_matrix, new_key
    opt_state = optimizer.init(params)
    CN=1
    print('Training started')
    rng_key = jax.random.key(keynum)
    t1 = time.time()
    # datas =[]
    # steps = 1500
    # print(f't1: {t1:.4e}')
    for i in range(steps):
        print(i)
        lambda_reg = 1e-3
        # params, opt_state, eloc, rng_key, CN, rank = step_minSR(params, rng_key, opt_state, CN, lambda_reg)
        params, opt_state, eloc, CN, rank = step(params, rng_key, opt_state, CN)
        # ranks.append(rank)
        # CNs.append(CN)
        rng_key, new_key = jax.random.split(rng_key)
        times.append(time.time() -t1)
        energies.append(jnp.mean(eloc))
        vars.append(jnp.var(jnp.abs(eloc)))
        CNs.append(CN)
        ranks.append(rank)
        if math.isnan(jnp.real(energies[-1])):
            print("NaN detected, stopping training.")
            break
        if i>3:
            if (times[-1]-times[1]) > TIME_LIMIT: #end after one hour of training
                print("Time limit exceeded, stopping training.")
                break
        # if i % 2000 == 300:
        #     print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
        #     print(f'Time Since Start: {times[-1]:.4e} seconds')
        #     E_min_sr = eloc_final(params,numsamples_f = 5*10**3)
        #     jnp.save(folder+f"/Data/energies-{opt}-s{i}-{Nx}.npy", jnp.array(energies))
        #     jnp.save(folder + f"/Data/vars-{opt}-s{i}-{Nx}.npy", jnp.array(vars))
        #     jnp.save(folder + f"/Data/times-{opt}-s{i}-{Nx}.npy", jnp.array(times))
        #     jnp.save(folder + f"/Data/ranks-{opt}-s{i}-{Nx}.npy", jnp.array(ranks))
        #     with open(folder + f'/Models/model_params-{opt}-s{i}-{Nx}.pkl', 'wb') as f:
        #         pickle.dump(params, f)
        #     plt.plot(jnp.log(jnp.array(vars)), label = f'{opt}, E/N:{E_min_sr[0]/N**2:.6f} +/- {E_min_sr[1]/N**2:.3e}')
        #     # plt.axline((0,E_min_sr[0]),(1,E_min_sr[0]), c='r', ls = '--')
        #     plt.legend()
        #     plt.savefig(folder + f'/Plots/{opt}-Heis-{steps}-s{i}-{Nx}.pdf')
        #     plt.close()
    return params,energies, vars , times, CNs, ranks

def get_grad(params, key, numsamples, Nx, Ny, model, CN):
    samples = model.apply(params,key,numsamples,Nx, Ny,method="sample")
    log_amps = model.apply(params,samples)
    e_loc = local_energy(samples, params, model, log_amps)
    e_loc_c = e_loc - e_loc.mean()
    grads,CN, rank = get_minSR_gradients(params, samples, e_loc_c,CN)
    # grads, S_matrix = get_minSR_gradients(params, samples, e_loc_c)

    return grads, e_loc, CN, rank
    # return grads, e_loc, S_matrix
J1 = 1
J2 = 0.5
def inverse_schedule(step):
    return lr/(1+step/step_decay)
optimizer = optax.sgd(learning_rate=inverse_schedule, momentum = m)
# optimizer = optax.adam(learning_rate = inverse_schedule)
key1, key2 = jax.random.split(jax.random.key(1))
x = jax.random.randint(key1, (2,4,4), 0, 2) # Dummy input data
# # print(x)
# params = model.init(key2, x) # Initialization call
x
pkl_path = f'./init-params/j1j2_params-{dh}.pkl'
with open(pkl_path, 'rb') as f:
    params = pickle.load(f)
numparams = sum([x.size for x in jax.tree.leaves(params)])
opt_state = optimizer.init(params)

rng_key = jax.random.key(0)
##      N = 4 ##################
# Nx = N-6
# Ny = N-6
# @partial(jit)
# def step(params, rng_key, opt_state, CN):
#     grads, e_loc, CN, rank = get_grad(params, rng_key, numsamples, Nx, Ny, model, CN)
#     # grads, e_loc, S_matrix = get_grad(params, new_key, numsamples, Nx, Ny, model)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, e_loc, CN, rank
#     # return params, opt_state, e_loc, S_matrix, new_key
energies = []
vars = []
times=[]
CNs=[]
ranks = []
# CN = 1
# print(f'Training started N={Nx}')
# t1 = time.time()
# for i in range(4000):
#     params, opt_state, eloc, CN, rank = step(params, rng_key, opt_state, CN)
#     # ranks.append(rank)
#     # CNs.append(CN)
#     rng_key, new_key = jax.random.split(rng_key)
#     times.append(time.time() -t1)
#     # params, opt_state, (_,eloc), rng_key = step_adam(params, rng_key, opt_state)
#     # params, opt_state, eloc, S_matrix, rng_key = step(params, rng_key, opt_state)
#     # if i % 10 == 0:
#     #   print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
#     if math.isnan(jnp.real(jnp.mean(eloc))):
#         print("NaN encountered at step ", i)
#         break
#     energies.append(jnp.mean(eloc))
#     vars.append(jnp.var(jnp.abs(eloc)))
#     if i % 2000 == 300:
#         print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
#         print(f'Time Since Start: {times[-1]:.4e} seconds')
#         E_min_sr = eloc_final(numsamples_f = 3*10**3)
#         jnp.save(folder+f"/Data/energies-{opt}-s{i}-{Nx}.npy", jnp.array(energies))
#         jnp.save(folder + f"/Data/vars-{opt}-s{i}-{Nx}.npy", jnp.array(vars))
#         jnp.save(folder + f"/Data/times-{opt}-s{i}-{Nx}.npy", jnp.array(times))
#         jnp.save(folder + f"/Data/ranks-{opt}-s{i}-{Nx}.npy", jnp.array(ranks))
#         with open(folder + f'/Models/model_params-{opt}-s{i}-{Nx}.pkl', 'wb') as f:
#             pickle.dump(params, f)
#         plt.plot(jnp.log(jnp.array(vars)), label = f'{opt}, E/N:{E_min_sr[0]/N**2:.6f} +/- {E_min_sr[1]/N**2:.3e}')
#         # plt.axline((0,E_min_sr[0]),(1,E_min_sr[0]), c='r', ls = '--')
#         plt.legend()
#         plt.savefig(folder + f'/Plots/{opt}-Heis-{steps}-s{i}-{Nx}.pdf')
#         plt.close()
    # grads.append(stats["grad"])
    # li.append(stats["norm_inv"])

    # pbar.set_postfix(loss = f'{jnp.log(vars[-1]):.4f},{energies[-1]:.3f}')
####

##      N = 6 ##################
energies = []
vars = []
times=[]
CNs=[]
ranks = []
Nx=N
Ny=N
for lr in [lr]:
    lr = lr
    for keynum in range(iter_num):
      params, energies, vars , times, CNs, ranks = train_minsr(params,steps, keynum=keynum, energies = energies, vars = vars, times = times, CNs = CNs, ranks = ranks)
      E_min_sr = eloc_final(params,numsamples_f=numsamples_f)
      min_sr_time = times[-1] - times[0]
      minsr_var = (E_min_sr[1]*jnp.sqrt(numsamples_f))**2
      v_score = N**2*minsr_var/(E_min_sr[0])**2
      with open(csv_file, "a") as file:
            writer = csv.writer(file)
            writer.writerow([N,numsamples, numparams, numsamples/numparams,v_score,E_min_sr[0]/N**2,E_min_sr[1]/N**2,len(energies),min_sr_time,lr,bool_tr,step_decay,keynum])

   

# Nx = N-4
# Ny = N-4
# @partial(jit)
# def step(params, rng_key, opt_state, CN):
#     grads, e_loc, CN, rank = get_grad(params, rng_key, numsamples, Nx, Ny, model, CN)
#     # grads, e_loc, S_matrix = get_grad(params, new_key, numsamples, Nx, Ny, model)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, e_loc, CN, rank
#     # return params, opt_state, e_loc, S_matrix, new_key
# CN = 1
# print(f'Training started N={Nx}')
# t1 = time.time()
# for i in range(5000):
#     params, opt_state, eloc, CN, rank = step(params, rng_key, opt_state, CN)
#     # ranks.append(rank)
#     # CNs.append(CN)
#     rng_key, new_key = jax.random.split(rng_key)
#     times.append(time.time() -t1)
#     # params, opt_state, (_,eloc), rng_key = step_adam(params, rng_key, opt_state)
#     # params, opt_state, eloc, S_matrix, rng_key = step(params, rng_key, opt_state)
#     # if i % 10 == 0:
#     #   print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
#     if math.isnan(jnp.real(jnp.mean(eloc))):
#         print("NaN encountered at step ", i)
#         break
#     energies.append(jnp.mean(eloc))
#     vars.append(jnp.var(jnp.abs(eloc)))
#     if i % 2000 == 300:
#         print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
#         print(f'Time Since Start: {times[-1]:.4e} seconds')
#         E_min_sr = eloc_final(numsamples_f = 3*10**3)
#         jnp.save(folder+f"/Data/energies-{opt}-s{i}-{Nx}.npy", jnp.array(energies))
#         jnp.save(folder + f"/Data/vars-{opt}-s{i}-{Nx}.npy", jnp.array(vars))
#         jnp.save(folder + f"/Data/times-{opt}-s{i}-{Nx}.npy", jnp.array(times))
#         jnp.save(folder + f"/Data/ranks-{opt}-s{i}-{Nx}.npy", jnp.array(ranks))
#         with open(folder + f'/Models/model_params-{opt}-s{i}-{Nx}.pkl', 'wb') as f:
#             pickle.dump(params, f)
#         plt.plot(jnp.log(jnp.array(vars)), label = f'{opt}, E/N:{E_min_sr[0]/N**2:.6f} +/- {E_min_sr[1]/N**2:.3e}')
#         # plt.axline((0,E_min_sr[0]),(1,E_min_sr[0]), c='r', ls = '--')
#         plt.legend()
#         plt.savefig(folder + f'/Plots/{opt}-Heis-{steps}-s{i}-{Nx}.pdf')
#         plt.close()
#     # grads.append(stats["grad"])
#     # li.append(stats["norm_inv"])

#     # pbar.set_postfix(loss = f'{jnp.log(vars[-1]):.4f},{energies[-1]:.3f}')
# ####

# ##      N = 8 ##################
# Nx = N-2
# Ny = N-2
# @partial(jit)
# def step(params, rng_key, opt_state, CN):
#     grads, e_loc, CN, rank = get_grad(params, rng_key, numsamples, Nx, Ny, model, CN)
#     # grads, e_loc, S_matrix = get_grad(params, new_key, numsamples, Nx, Ny, model)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, e_loc, CN, rank
#     # return params, opt_state, e_loc, S_matrix, new_key
# CN = 1
# print(f'Training started N={Nx}')
# t1 = time.time()
# for i in range(10000):
#     params, opt_state, eloc, CN, rank = step(params, rng_key, opt_state, CN)
#     # ranks.append(rank)
#     # CNs.append(CN)
#     rng_key, new_key = jax.random.split(rng_key)
#     times.append(time.time() -t1)
#     # params, opt_state, (_,eloc), rng_key = step_adam(params, rng_key, opt_state)
#     # params, opt_state, eloc, S_matrix, rng_key = step(params, rng_key, opt_state)
#     # if i % 10 == 0:
#     #   print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
#     if math.isnan(jnp.real(jnp.mean(eloc))):
#         print("NaN encountered at step ", i)
#         break
#     energies.append(jnp.mean(eloc))
#     vars.append(jnp.var(jnp.abs(eloc)))
#     if i % 2000 == 300:
#         print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
#         print(f'Time Since Start: {times[-1]:.4e} seconds')
#         E_min_sr = eloc_final(numsamples_f = 6*10**3)
#         jnp.save(folder+f"/Data/energies-{opt}-s{i}-{Nx}.npy", jnp.array(energies))
#         jnp.save(folder + f"/Data/vars-{opt}-s{i}-{Nx}.npy", jnp.array(vars))
#         jnp.save(folder + f"/Data/times-{opt}-s{i}-{Nx}.npy", jnp.array(times))
#         jnp.save(folder + f"/Data/ranks-{opt}-s{i}-{Nx}.npy", jnp.array(ranks))
#         with open(folder + f'/Models/model_params-{opt}-s{i}-{Nx}.pkl', 'wb') as f:
#             pickle.dump(params, f)
#         plt.plot(jnp.log(jnp.array(vars)), label = f'{opt}, E/N:{E_min_sr[0]/N**2:.6f} +/- {E_min_sr[1]/N**2:.3e}')
#         # plt.axline((0,E_min_sr[0]),(1,E_min_sr[0]), c='r', ls = '--')
#         plt.legend()
#         plt.savefig(folder + f'/Plots/{opt}-Heis-{steps}-s{i}-{Nx}.pdf')
#         plt.close()
#     # grads.append(stats["grad"])
#     # li.append(stats["norm_inv"])

#     # pbar.set_postfix(loss = f'{jnp.log(vars[-1]):.4f},{energies[-1]:.3f}')
# ####

# ##. N=10 #####
# Nx = N
# Ny = N
# rng_key = jax.random.key(1)
# @partial(jit)
# def step(params, rng_key, opt_state, CN):
#     grads, e_loc, CN, rank = get_grad(params, rng_key, numsamples, Nx, Ny, model, CN)
#     # grads, e_loc, S_matrix = get_grad(params, new_key, numsamples, Nx, Ny, model)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, e_loc, CN, rank
#     # return params, opt_state, e_loc, S_matrix, new_key
# CN = 1
# print(f'Training started N={Nx}')
# t1 = time.time()
# for i in range(steps):
#     params, opt_state, eloc, CN, rank = step(params, rng_key, opt_state, CN)
#     # ranks.append(rank)
#     # CNs.append(CN)
#     rng_key, new_key = jax.random.split(rng_key)
#     times.append(time.time() -t1)
#     # params, opt_state, (_,eloc), rng_key = step_adam(params, rng_key, opt_state)
#     # params, opt_state, eloc, S_matrix, rng_key = step(params, rng_key, opt_state)
#     # if i % 10 == 0:
#     #   print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
#     if math.isnan(jnp.real(jnp.mean(eloc))):
#         print("NaN encountered at step ", i)
#         break
#     if times[-1] > TIME_LIMIT:
#         print("Time limit exceeded at step ", i)
#         break
#     energies.append(jnp.mean(eloc))
#     vars.append(jnp.var(jnp.abs(eloc)))
#     if i % 2000 == 300:
#         print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
#         print(f'Time Since Start: {times[-1]:.4e} seconds')
#         E_min_sr = eloc_final(numsamples_f = 5*10**3)
#         jnp.save(folder+f"/Data/energies-{opt}-s{i}-{Nx}.npy", jnp.array(energies))
#         jnp.save(folder + f"/Data/vars-{opt}-s{i}-{Nx}.npy", jnp.array(vars))
#         jnp.save(folder + f"/Data/times-{opt}-s{i}-{Nx}.npy", jnp.array(times))
#         jnp.save(folder + f"/Data/ranks-{opt}-s{i}-{Nx}.npy", jnp.array(ranks))
#         with open(folder + f'/Models/model_params-{opt}-s{i}-{Nx}.pkl', 'wb') as f:
#             pickle.dump(params, f)
#         plt.plot(jnp.log(jnp.array(vars)), label = f'{opt}, E/N:{E_min_sr[0]/N**2:.6f} +/- {E_min_sr[1]/N**2:.3e}')
#         # plt.axline((0,E_min_sr[0]),(1,E_min_sr[0]), c='r', ls = '--')
#         plt.legend()
#         plt.savefig(folder + f'/Plots/{opt}-Heis-{steps}-s{i}-{k}.pdf')
#         plt.close()
#     # grads.append(stats["grad"])
#     # li.append(stats["norm_inv"])

#     # pbar.set_postfix(loss = f'{jnp.log(vars[-1]):.4f},{energies[-1]:.3f}')
# with open(folder + f'/model_params-{steps}-{opt}.pkl', 'wb') as f:
#   pickle.dump(params, f)


# E_min_sr = eloc_final(params,numsamples_f=numsamples_f)

# @partial(jit, static_argnums=(3,))
# def step_adam(params, rng_key, opt_state, get_loss=get_loss):
#     value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, rng_key, numsamples, Nx, Ny, model)
#     # grads_flat, _ = jax.tree_util.tree_flatten(grads)
#     # global_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
#     # g_factor = jnp.minimum(1.0, grad_clip_norm / global_l2)
#     # grads = jax.tree_util.tree_map(lambda g: g * g_factor, grads)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, value

# def inverse_schedule(step):
#     return lr/(1+step/5000)
# optimizer = optax.sgd(learning_rate=inverse_schedule,momentum = 0.7)
# # optimizer = optax.adam(learning_rate = 2e-4)
# key1, key2 = jax.random.split(jax.random.key(1))
# # x = jax.random.randint(key1, (2,4,4), 0, 2) # Dummy input data
# # print(x)
# # params = model.init(key2, x) # Initialization call
# params = params0 # same starting point 
# opt_state = optimizer.init(params)
# Nx = N
# Ny = N
# rng_key = jax.random.key(1)


# energies = []
# vars = []
# times=[]
# print('Training started')
# t1 = time.time()
# for i in range(steps):
#     params, opt_state, eloc, CN = step(params, rng_key, opt_state, CN)
#     rng_key, new_key = jax.random.split(rng_key)
#     # params, opt_state, (_,eloc)= step_adam(params, rng_key, opt_state)
#     # params, opt_state, eloc, S_matrix, rng_key = step(params, rng_key, opt_state)
#     # if i % 10 == 0:
#     #   print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))

#     energies.append(jnp.mean(eloc))
#     vars.append(jnp.var(eloc))
#     times.append(time.time()-t1)
#     # grads.append(stats["grad"])
#     # li.append(stats["norm_inv"])

#     # pbar.set_postfix(loss = f'{jnp.log(vars[-1]):.4f},{energies[-1]:.3f}')

# with open(folder + f'/Models/model_params-{steps}-Adam.pkl', 'wb') as f:
#   pickle.dump(params, f)

# adam_vars = jnp.real(jnp.array(vars))
# adam_energies = jnp.real(jnp.array(energies))
# adam_energies_i = jnp.imag(jnp.array(energies))
# t2 = time.time()
# adam_time = t2 -t1
# E_adam = eloc_final()


# plt.plot(jnp.array(min_sr_vars), label = f'{opt}, time: {min_sr_time:.4f}, Energy = {E_min_sr[0]:.6f} +/- {E_min_sr[1]:.3e}')
# plt.plot(jnp.array(min_sr_vars), label = f'{opt}, time: {t2-t1:.4f}, Energy = {E_min_sr[0]:.6f} +/- {E_min_sr[1]:.3e}')
# plt.yscale('log')
# plt.plot(jnp.log(jnp.array(adam_vars)), label = f'Adam, time: {adam_time}')
##   plt.plot(jnp.log(jnp.array(e2)), label = 'Adam')
# plt.legend()
# plt.title(f'J1/J2 N={N}x{N} Vars')
# plt.xlabel('Iterations')
# plt.ylabel('log(var)')
# plt.savefig(folder + f'/Plots/J1J2-{steps}-vars.pdf')
# plt.close()

# plt.plot(min_sr_energies[300:], label = f'{opt}, E/N = {E_min_sr[0]/N**2:.6f} +/- {E_min_sr[1]/N**2:.2e}')
# plt.axline((0,E_min_sr[0]),(1,E_min_sr[0]), c='r', ls = '--')
# plt.plot(min_sr_energies, label = f'{opt}, time: {t2-t1:.4f}, Energy = {E_min_sr[0]:.4f} +/- {E_min_sr[1]:.3e}')
# plt.axline((0,E_min_sr[0]),(1,E_min_sr[0]), c='g', ls = '--')
#plt.plot(jnp.log(jnp.array(adam_vars)), label = f'Adam, time: {adam_time}')
##   plt.plot(jnp.log(jnp.array(e2)), label = 'Adam')

# plt.legend()
# plt.title(f'J1/J2 N={N} E')
# plt.xlabel('Iterations')
# plt.ylabel('energies')
# plt.savefig(folder + f'/Plots/J1J2-{steps}-energies.pdf')

# jnp.save(folder+ f"/Data/energies-{opt}.npy", min_sr_energies)
# jnp.save(folder + f"/Data/vars-{opt}.npy", min_sr_vars)
# jnp.save(folder + f"/Data/CNs-{opt}.npy", min_sr_CNs)
# jnp.save(folder + f"/Data/times-{opt}.npy", times)
# jnp.save(folder + f"/Data/ranks-{opt}.npy", jnp.array(ranks))
# jnp.save(folder+f"/Data/energies-adam.npy", adam_energies)
# jnp.save(folder + f"/Data/vars-adam.npy", adam_vars)
# jnp.save(folder + f"/Data/times-adam.npy", times)

