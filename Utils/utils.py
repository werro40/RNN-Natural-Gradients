import jax
# from RNN2D import StackedRNNModel
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple, Union, Optional, Callable, Any
from jax import jit
from math import ceil
jax.config.update("jax_enable_x64", True)
jax_dtype = jnp.float64
import matplotlib.pyplot as plt
import json
# import config
# N,steps,dh,lr,momentum,title, numsamples, jax_dtype =  config.N, config.steps, config.dh, config.lr, config.momentum, config.title, config.numsamples, config.jax_dtype
# Nx = N
# Ny = N
class data_class: 
    def __init__(self, **kwargs): 
        # Loop through the key-value pairs passed in via kwargs 
        # Store the keys in a list to maintain order during updates
        self._keys = list(kwargs.keys())
        
        for key in self._keys: 
            setattr(self, key, []) 

    def update(self, args):
        # Ensure args is the same length as the number of keys
        if len(args) != len(self._keys):
            raise ValueError(f"Expected {len(self._keys)} arguments, got {len(args)}")
            
        # Enumerate through the tracked keys to match them with the args index
        for idx, key in enumerate(self._keys): 
            # Get the list attribute dynamically and append the item
            getattr(self, key).append(args[idx])

    def save(self, folder,metadata={}, opt='minsr',i=0): 
        # Ensure the directory exists before saving
        # os.makedirs(os.path.join(folder, 'data'), exist_ok=True)
        
        # Loop through only the data keys we initialized
        # for key in self._keys: 
        #     array = getattr(self, key)
        #     # Convert to numpy array and save
        #     steps = len(array)
        #     jnp.save(folder +'/Data/'+key+f'-{opt}-s{steps}-{i}.npy', jnp.array(array))
        data_to_save = {key: jnp.array(getattr(self, key)) for key in self._keys}
        data_to_save['metadata'] = metadata  # Save metadata as a JSON string  ']
        key1 = self._keys[0]
        step = len(getattr(self,key1))
        jnp.savez(folder+f'/Data/data-{opt}-s{step}-{i}.npz', **data_to_save)

    def plot(self,key,folder, opt='minsr',i=0, log=False, **kwargs):
        x = jnp.array(getattr(self, key))
        steps = len(x)
        fig, ax = plt.subplots(dpi=100)
        ax.plot(x, **kwargs)
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel(key, fontsize=16)
        ax.grid()
        plt.legend()
        if log:
            ax.set_yscale('log')
        plt.savefig(folder+'/Plots/'+key+f'-{opt}-s{steps}-{i}.pdf')
        plt.close()
def slurm_time_to_seconds(t):
    days = 0
    if '-' in t:
        days, t = t.split('-')
        days = int(days)
    parts = [int(p) for p in t.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)  # pad in case format is just MM:SS
    hours, minutes, seconds = parts
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

def local_energy_h2d(samples, params, model, log_psi) -> List[float]:

    """Computes the local energy of the 2D Heisenberg model"""

    numsamples,Nx,Ny = samples.shape

    N = Nx*Ny

    local_energies = jnp.zeros((numsamples), dtype = jax_dtype)
    local_energies += jnp.sum(0.25*(2*samples[:,:-1,:]-1)*(2*samples[:,1:,:]-1), axis = (1,2)) #diagonal elements (right neighbours)
    local_energies += jnp.sum(0.25*(2*samples[:,:,:-1]-1)*(2*samples[:,:,1:]-1), axis = (1,2)) #diagonal elements (down neighbours)

    # for i in range(Nx-1): #diagonal elements (right neighbours)

    #     spins_products = 0.25*(2*samples[:,i]-1)*(2*samples[:,i+1]-1)
    #     local_energies += jnp.sum(jnp.copy(spins_products), axis = 1)

    # for j in range(Ny-1): #diagonal elements (upward neighbours (or downward, it depends on the way you see the lattice))

    #     spins_products = 0.25*(2*samples[:,:,j]-1)*(2*samples[:,:,j+1]-1)
    #     local_energies += jnp.sum(jnp.copy(spins_products), axis = 1)

    def step_fn_horizontal(n, state):

        s, output = state
        _, Nx,Ny = s.shape

        i = (n//Ny) #set back to zero when equal to Nx-1
        j = n%Ny

        flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
        flipped_state = flipped_state.at[:, i+1,j].set(1 - flipped_state[:, i+1,j])
        flipped_logpsi = 0.5*model.apply(params,flipped_state)#,method="logprobs_c4vsym")
        output += (s[:, i,j] + s[:, i+1,j] == 1) *(-0.5)* jnp.exp(flipped_logpsi - log_psi)

        return s, output



    def step_fn_vertical(n, state):

        s, output = state
        _, Nx,Ny = s.shape

        j = (n//Nx) #set back to zero when equal to Nx-1
        i = n%Nx


        flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
        flipped_state = flipped_state.at[:, i,j+1].set(1 - flipped_state[:, i,j+1])
        flipped_logpsi = 0.5*model.apply(params,flipped_state)#,method="logprobs_c4vsym")
        output += ((s[:, i,j] + s[:, i,j+1] == 1)*(-0.5))*jnp.exp(flipped_logpsi - log_psi)

        return s, output

    # Off Diagonal Term
    output = jnp.zeros((numsamples), dtype=jax_dtype)
    _, off_diag_term_vertical = jax.lax.fori_loop(0, Nx*(Ny-1), step_fn_vertical, (samples, output))
    _, off_diag_term_horizontal = jax.lax.fori_loop(0, (Nx-1)*(Ny), step_fn_horizontal, (samples, output))

    local_energies += off_diag_term_vertical +  off_diag_term_horizontal

    return local_energies
    

def local_energy_j1j2(samples, params, model, log_psi) -> List[float]:

    """Computes the local energy of the 2D Heisenberg model"""

    numsamples,Nx,Ny = samples.shape

    N = Nx*Ny

    local_energies = jnp.zeros((numsamples), dtype = jax_dtype)

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

def local_energy_tfim(samples, params, model, log_psi) -> List[float]:
    """Computes the local energy of the system"""
    numsamples,N = samples.shape

   # Interaction Term
    interaction_term = - jnp.sum((2*samples-1) * (2*jnp.roll(samples, 1, axis = 1)-1), axis=1) #nearest neighbor interactoui
    interaction_term += (2*samples[:,0]-1) * (2*samples[:,-1]-1) #to impose open boundary conditions (OBC)

    def step_fn_transverse(i, state):
        s, output = state #Why are we flipping the state? because sigma_x is the flip gate!
        flipped_state = s.at[:, i].set(1 - s[:, i])
        flipped_logpsi = 0.5*model.apply(params,flipped_state)
        output += - jnp.exp(flipped_logpsi - log_psi)
        return s, output

    # Off Diagonal Term
    output = jnp.zeros((numsamples), dtype=jnp.float64)
    _, off_diag_term = jax.lax.fori_loop(0, N, step_fn_transverse, (samples, output))


    loc_e = interaction_term + off_diag_term

    return loc_e

def local_energy_cluster(samples, params, model, log_psi) -> List[complex]:
    """Computes the local energy of the system"""
    NUMBER_OF_SAMPLES, N = samples.shape

    spins = 2 * samples - 1

    def step_fn_cluster(i, state):
        s, output = state #Why are we flipping the state? because sigma_x is the flip gate!
        # flipped_state = s.at[:, i].set(1-s[:, i]) #debug
        flipped_state = s.at[:, i-1].set(1-s[:, i-1])
        flipped_state = flipped_state.at[:, i+1].set(1-flipped_state[:, i+1])
        flipped_logpsi = model.apply(params,flipped_state)
        output += -(1-2*flipped_state[:, i])*jnp.exp(flipped_logpsi - log_psi)  #-flipped_state[:, i] is for Z_i term
        return s, output

    # Off Diagonal Term
    output = jnp.zeros((NUMBER_OF_SAMPLES), dtype=jnp.complex128)
    _, off_diag_term = jax.lax.fori_loop(2-1, N-2, step_fn_cluster, (samples, output))

    flipped_state = samples.at[:, 1].set(1-samples[:, 1])
    flipped_logpsi = model.apply(params,flipped_state)
    off_diag_term += -(1-2*flipped_state[:,0])*jnp.exp(flipped_logpsi - log_psi)

    flipped_state = samples.at[:, N-2].set(1-samples[:, N-2])
    flipped_state = flipped_state.at[:, N-1].set(1-flipped_state[:, N-1])
    flipped_logpsi = model.apply(params,flipped_state)
    off_diag_term += -jnp.exp(flipped_logpsi - log_psi)

    flipped_state = samples.at[:, N-3].set(1-samples[:, N-3])
    flipped_logpsi = model.apply(params,flipped_state)
    off_diag_term += -(1-2*flipped_state[:,N-2])*(1-2*flipped_state[:,N-1])*jnp.exp(flipped_logpsi - log_psi)

    loc_e = off_diag_term

    return loc_e



def _apply_step(params, dtheta_tree, step_size): 
    """params_new = params - step_size * dtheta_tree""" 
    return jax.tree.map(lambda p, d: p - step_size * d, params, dtheta_tree)
def _flatten_jacobian(jacobian, numsamples):
    """Flatten a pytree-valued jacobian into a single (numsamples, nparams) matrix,
    plus the metadata needed to unflatten a parameter-shaped vector back into the
    original pytree structure."""
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

def _flatten_params(params):
    """Flatten a pytree-valued jacobian into a single (numsamples, nparams) matrix,
    plus the metadata needed to unflatten a parameter-shaped vector back into the
    original pytree structure."""
    flattened_params, tree = jax.tree_util.tree_flatten(params)
    shapes = [it.shape for it in flattened_params]
    sizes = [it[0].size for it in flattened_params]
    slices = []
    last = 0
    for s in sizes:
        slices.append(slice(last, last + s))
        last += s
    jac = jnp.concatenate([it.reshape(it.shape[0], -1) for it in flattened_jac], axis=-1)
    return jac, tree, shapes, slices

def _unflatten_like_params(flat_vec, tree, shapes, slices):
    """Inverse of _flatten_jacobian for a single parameter-shaped vector (not a jacobian)."""
    flat_tree = []
    for shape, _slice in zip(shapes, slices):
        flat_tree.append(flat_vec[_slice].reshape(shape[1:]))
    return jax.tree_util.tree_unflatten(tree, flat_tree)


def _apply_step(params, dtheta_tree, step_size):
    """params_new = params - step_size * dtheta_tree"""
    return jax.tree.map(lambda p, d: p - step_size * d, params, dtheta_tree)
# def get_loss(params, key, numsamples, Nx, Ny, model):

#     samples = model.apply(params,key, numsamples,Nx,Ny, method="sample")
#     log_probs = model.apply(params,samples)#),method="logprobs_c4vsym")

#     e_loc = jax.lax.stop_gradient(local_energy(samples, params, model, 0.5*log_probs))
#     e_avg = e_loc.mean()

#     loss = jnp.mean(jnp.multiply(log_probs, e_loc) - jnp.multiply(e_avg, log_probs))
#     return loss, e_loc
# #param_count = sum(x.size for x in jax.tree_leaves(params))


    # return grads, e_loc, S_matrix
#     # return grads, e_loc, S_matrix
# def get_SPRING_gradients(params, samples, local_energies, gradients): # editted fic later, remove S_pc
#   jacobian = jax.jacrev(log_probs_fun)(params, samples)

#   numsamples = samples.shape[0]

#   flattened_jac, tree = jax.tree_util.tree_flatten(jacobian)

#   shapes = [it.shape for it in flattened_jac]

#   slices = []
#   last = flattened_jac[0][0].size
#   slices.append(slice(0,last))
#   for it in flattened_jac[1:]:
#       slices.append(slice(last,last+it[0].size))
#       last += it[0].size

#   jac = jnp.concatenate([it.reshape(it.shape[0],-1) for it in flattened_jac], axis=-1)
#   jac -= jnp.mean(jac, axis = 0)
#   jac = jac/ jnp.sqrt(numsamples)
#   XdaggerX = jac @ jac.T

#   try: gradients
#   except: gradients = jnp.zeros()

#   # eigvals, _ = jnp.linalg.eigh(XdaggerX)
#   # max_eigval = max(eigvals)

#   norm =  jnp.sqrt(jnp.trace(XdaggerX @ XdaggerX))

# #   step_decay = 1#/(1+norm/1e7)
#   # we can do the same with lambda
#   lambda_reg = 2e-3

#   zeta = local_energies - mu * jac @ gradients

#   XdaggerX_inv = jax.scipy.linalg.inv( XdaggerX + lambda_reg * jnp.eye(XdaggerX.shape[0]) + 1/numsamples *jnp.ones( (XdaggerX.shape[0],XdaggerX.shape[0]) ) )
#   gradients =  (jac.T @ XdaggerX_inv @ zeta * ( 2 / jnp.sqrt(numsamples)) + mu * gradients)


#   # diag = jnp.diag(XdaggerX)
#   # norms = jnp.sqrt(jnp.outer(diag, diag))

#   # # elementwise division
#   # S_pc = XdaggerX / norms


#   # S_pc_inv = jax.scipy.linalg.inv((S_pc + lmbda * jnp.eye(XdaggerX.shape[0])))
#   # local_energies_pc  = local_energies/jnp.sqrt(diag)

#   # gradients = (jac.T/jnp.sqrt(diag)) @ S_pc_inv @ local_energies_pc * ( 2 / jnp.sqrt(numsamples))

#   ### unflatten
#   flat_tree = []
#   for shape, _slice in zip(shapes, slices):
#       flat_tree.append(gradients[_slice].reshape(shape[1:]))

#   original_grad = jax.tree_util.tree_unflatten(tree, flat_tree)

#   return original_grad, gradients

# def get_spring_grad(params, key, numsamples, Nx, Ny, model, gradients):
#     samples = model.apply(params,key,numsamples,Nx, Ny,method="sample") # This line with the next one take ~18.62it/s for N = 20 1DTFIM
#     log_probs = model.apply(params,samples)#,method="logprobs_c4vsym")
#     e_loc = local_energy(samples, params, model, 0.5*log_probs)
#     e_loc_c = e_loc - e_loc.mean()
#     grads, gradients = get_SPRING_gradients(params, samples, e_loc_c, gradients)
#     return grads, e_loc, gradients

# mu = 0.6436
# def recursive_items(dictionary, current_path=None):
#     if current_path is None:
#         current_path = []
#     for key, value in dictionary.items():
#         new_path = current_path + [key]
#         if isinstance(value, dict):
#             yield from recursive_items(value, new_path)
#         else:
#             yield new_path, value


def final_energy(params, key, model, Nx, Ny, num_samples_final):
  samples = model.apply(params, key, num_samples_final, Nx, Ny, method="sample")
  log_probs = model.apply(params,samples)
  e_loc = local_energy(samples, params, model, 0.5*log_probs)#, offdiag_logpsi, 0.5*log_probs)
  return e_loc
