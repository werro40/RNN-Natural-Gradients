

#Class defines the wavefunction, should eventually be put in another file.
# Using Jax-Flax-Linen to build RNN wave function for the 1D Transverse Ising model
from functools import partial
import jax
# import seaborn as sns
import time

# from Heisenberg2D import CNs
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Tuple, Union, Optional, Callable, Any
import optax
# from tqdm import tqdm
import pickle
from jax import jit
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import datetime
from Models.tfim_rnn import RNNModel
from Hamiltonians.hamiltonians import local_energy_tfim as local_energy_sequential
test =False
params_path = './init-params/tfim_params-32.pkl'
opt_dict = {  'title': '-adam-run',
                'opt': 'adam',
                'trust_region': False,
                 'CN-1': False,
                 'CN-2': False,
                 'lr': 0.05,
                 'm': 0.7,
                 'step_decay': 1000,
                 }

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
        data_to_save['metadata'] = metadata
        key1 = self._keys[0]
        step = len(getattr(self,key1))
        jnp.savez(folder+f'/Data/data-{opt}-s{step}-{i}.npz', **data_to_save)

    def plot(self,key,folder, opt='minsr',i=0, log=False, **kwargs):
        x = jnp.abs(jnp.array(getattr(self, key)))
        steps = len(x)
        fig, ax = plt.subplots(dpi=100)
        ax.plot(x, **kwargs)
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel(key, fontsize=16)
        ax.grid()
        plt.legend()
        if log:
            ax.set_yscale('log')
        plt.tight_layout()
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

if test == True:
  N = 2
  dh = 1
  steps = int(4e0)
  final_numsamples =3
  numsamples = 5

else:
  N = 200
  batch_size = 1
  samples_per_batch = 100
  numsamples = 500
  dh = 32
  steps = int(1e4)
  final_numsamples =3*10**4
  checkpoint_numsamples = 3000

# nls = [1,2,3,4,5]
# nss = [350,400,450]
# lrs = 0.03

d_model = 2
symmetries = False
# args = {'method': 'logprobs_c4vsym'} if symmetries else {}

n_layers = 1

title = opt_dict['title']
lr = opt_dict['lr']
m=opt_dict['m']
opt=opt_dict['opt']
step_decay = opt_dict['step_decay']
# CN_2 = int(opt_dict['CN-2'])
# CN_1 = int(opt_dict['CN-1'])
# trust_region = int(opt_dict['trust_region'])
# TIME_LIMIT = opt_dict['TIME_LIMIT']
# lr = 5e-4

s = f''' 
Bismillah-ir-Rahman-ir-Rahim
Model: 2D Heisenberg 
Title: {title}
Optimizer: {opt}
lambda_reg = 2e-3 constant
N = {N}
numsamples = {numsamples}
dh = {dh}
lr = {lr}/(1+step/{step_decay})
momentum = {m}
'''
if test:
  folder = f'./Experiments/TFIM-1D/TFIM-' + str(datetime.date.today()) +f'-test'
else:
  folder = f'./Experiments/TFIM-1D/TFIM-' + str(datetime.date.today()) +f'-{title}'

if not os.path.exists(folder):
  os.mkdir(folder)
  os.mkdir(folder + '/Data')
  os.mkdir(folder + '/Plots')
  os.mkdir(folder + '/Model')

s = f'''  
Bismillah-ir-Rahman-ir-Rahim
Model: 1D TFIM {title}
Optimizer: adam + minsr
Symmetry: None
lambda_reg = 2e-3
total_steps = {steps}
num_particles = N
hidden_dim = {dh}
step_reg = None
N = {N}
numsamples = {numsamples}
dh = {dh}
lr = 5.5e-3 (adam), variable (minSR)
'''

with open(folder + '/docs.txt','w') as f:
    f.write(s)

with open(params_path, 'rb') as f:
   init_params = pickle.load(f)


def get_loss(params, key, numsamples, N, model):

    samples = model.apply(params,key, numsamples,N, method="sample")
    log_probs = model.apply(params,samples)

    e_loc = jax.lax.stop_gradient(local_energy_sequential(samples, params, model, 0.5*log_probs))
    e_avg = e_loc.mean()

    loss = jnp.mean(jnp.multiply(log_probs, e_loc) - jnp.multiply(e_avg, log_probs))
    return loss, e_loc

def get_grad(params, key, numsamples, N, model):
    samples = model.apply(params,key,numsamples,N,method="sample") # This line with the next one take ~18.62it/s for N = 20 1DTFIM
    log_probs = model.apply(params,samples)
    e_loc = local_energy_sequential(samples, params, model, 0.5*log_probs)
    e_loc_c = e_loc - e_loc.mean()
    grads, eigs, rank = get_minSR_gradients(params, samples, e_loc_c)
    return grads, e_loc, eigs, rank

def log_probs_fun(params, samples):
    return 0.5*model.apply(params,samples)

def get_minSR_gradients(params, samples, local_energies): # editted fic later, remove S_pc
  jacobian = jax.jacrev(log_probs_fun)(params, samples)

  numsamples = samples.shape[0]

  flattened_jac, tree = jax.tree_util.tree_flatten(jacobian)

  shapes = [it.shape for it in flattened_jac]

  slices = []
  last = flattened_jac[0][0].size
  slices.append(slice(0,last))
  for it in flattened_jac[1:]:
      slices.append(slice(last,last+it[0].size))
      last += it[0].size

  jac = jnp.concatenate([it.reshape(it.shape[0],-1) for it in flattened_jac], axis=-1)
  jac -= jnp.mean(jac, axis = 0)
  jac = jac/ jnp.sqrt(numsamples)
  XdaggerX = jac @ jac.T

  eigs = jnp.linalg.eigvals(XdaggerX)
  rank = jnp.linalg.matrix_rank(XdaggerX)

  XdaggerX_inv = jax.scipy.linalg.inv((XdaggerX + lmbda * jnp.eye(XdaggerX.shape[0])))
  gradients = jac.T @ XdaggerX_inv @ local_energies * ( 2 / jnp.sqrt(numsamples))


  # diag = jnp.diag(XdaggerX)
  # norms = jnp.sqrt(jnp.outer(diag, diag))

  # # # elementwise division
  # S_pc = XdaggerX / norms


  # S_pc_inv = jax.scipy.linalg.inv((S_pc + lmbda * jnp.eye(XdaggerX.shape[0])))
  # local_energies_pc  = local_energies/np.sqrt(diag)

  # gradients = jac.T @ S_pc_inv @ local_energies_pc * ( 2 / jnp.sqrt(numsamples))

  ### unflatten
  flat_tree = []
  for shape, _slice in zip(shapes, slices):
      flat_tree.append(gradients[_slice].reshape(shape[1:]))

  original_grad = jax.tree_util.tree_unflatten(tree, flat_tree)

  return original_grad, eigs, rank

def inverse_schedule(step):
     return lr/(1+step/step_decay)

def minSR(N):

  # times = []
  # min_sr_energies = []  
  # min_sr_vars = []
  key = jax.random.key(42)
  x = jax.random.randint(jax.random.key(2), (5,N), 0, 2) # Dummy input data
  key, keyOld = jax.random.split(key)
  params = init_params
  optimizer = optax.sgd(learning_rate=inverse_schedule)
  opt_state = optimizer.init(params)

  # missing lr info
  # t1 = time.time()
  rng_key = jax.random.key(1)
  energies = []
  vars = []
  times = []
  ranks = []
  eigs_list = []
  t1 = time.time()
  print('Training started')
  for i in range(steps):
  # for i in tqdm(range(1000), desc="Epochs"):
      # params, opt_state, (_, eloc), rng_key = step(params, rng_key, opt_state)
      params, opt_state, eloc, rng_key, eigs, rank = step_minSR(params, rng_key, opt_state)
      # if i % 200 == 0:
      #   print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
      energies.append(jnp.mean(eloc))
      vars.append(jnp.var(eloc))
      times.append(time.time()-t1)
      eigs_list.append(eigs)
      ranks.append(rank)

#   final_numsamples =3*10**4
  final_samples = model.apply(params,rng_key,final_numsamples,N,method="sample")
  log_probs = model.apply(params,final_samples)#, method="logprobs_c4vsym")
  e_loc_final = local_energy_sequential(final_samples, params, model, 0.5*log_probs)
  e_loc_final_mean = jnp.mean(e_loc_final)
  e_loc_final_error = jnp.var(e_loc_final)/jnp.sqrt(final_numsamples)
  # t2 = time.time()
  # times.append(t2-t1)
  # min_sr_energies.append(energies)
  # min_sr_vars.append(vars)
  # time = t2 - t1
  jnp.save(folder+f"/Data/energies-minSR-{numsamples}.npy", energies)
  jnp.save(folder + f"/Data/vars-minSR-{numsamples}.npy", vars)
  jnp.save(folder + f"/Data/times-minSR-{numsamples}.npy", jnp.array(times))
  jnp.save(folder + f"/Data/ranks-minSR.npy", jnp.array(ranks))
  jnp.save(folder + f"/Data/eigs-minSR.npy", jnp.array(eigs_list))
  out = {'energies': energies, 'vars': vars, 'times': times, 'final_energy': e_loc_final_mean, 'final_error': e_loc_final_error, model: params}
  e_f = (e_loc_final_mean, e_loc_final_error)
  return e_f, vars, times

def ADAM(N):
  # times = []
  # adam_energies = []
  # adam_vars = []
  key = jax.random.key(43)
  x = jax.random.randint(jax.random.key(2), (5,N), 0, 2) # Dummy input data
  key, keyOld = jax.random.split(key)
  params = init_params
  # optimizer = optax.adam(learning_rate=inverse_schedule)
  optimizer = optax.adam(learning_rate=lr)


  opt_state = optimizer.init(params)
  # N = n
  # lr = 5

  # t1 = time.time()
  rng_key = jax.random.key(1)
  energies = []
  vars = []
  times = []
  print('Training started')
  for j in range(steps):
  # for i in tqdm(range(1000), desc="Epochs"):
      params, opt_state, (_, eloc), rng_key = step_adam(params, rng_key, opt_state)
      # params, opt_state, eloc, rng_key = step(params, rng_key, opt_state)
      # if i % 200 == 0:
      #   print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
      energies.append(jnp.mean(eloc))
      vars.append(jnp.var(eloc))
      times.append(time.time()-t1)
  final_samples = model.apply(params,rng_key,final_numsamples,N,method="sample")
  log_probs = model.apply(params,final_samples)#, method="logprobs_c4vsym")
  e_loc_final = local_energy_sequential(final_samples, params, model, 0.5*log_probs)
  e_loc_final_mean = jnp.mean(e_loc_final)
  e_loc_final_error = jnp.var(e_loc_final)/jnp.sqrt(final_numsamples)
  # t2 = time.time()
  jnp.save(folder + f"/Data/energies-adam-{numsamples}.npy", energies)
  jnp.save(folder + f"/Data/vars-adam-{numsamples}.npy", vars)
  jnp.save(folder + f"/Data/times-adam-{numsamples}.npy", jnp.array(times))
  with open(folder + f"/Model/adam-final-params-{numsamples}.pkl", 'wb') as f:
    pickle.dump(params, f)
#   out = {'energies': energies, 'vars': vars, 'times': times, 'final_energy': e_loc_fi
  # time = t2-t1
  # times.append(t2-t1)
  # adam_energies.append(jnp.array(energies))
  # adam_vars.append(jnp.array(vars))
  ef = (e_loc_final_mean, e_loc_final_error)
  return ef, vars, times

model = RNNModel(output_dim = 2, num_hidden_units = dh, RNNcell_type = "GRU")

#MIN SR in different particle numbers



@partial(jit, static_argnums=(3,))
def step_adam(params, rng_key, opt_state, get_grad=get_grad):
    rng_key, new_key = jax.random.split(rng_key)
    e_loc, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, numsamples, N, model)
    # grads, e_loc = get_grad(params, new_key, numsamples, N, model)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, e_loc, new_key#, value
#params 
@partial(jit, static_argnums=(3,))
def step_minSR(params, rng_key, opt_state, get_grad=get_grad):
    rng_key, new_key = jax.random.split(rng_key)
    # e_loc, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, numsamples, N, model)
    grads, e_loc, eigs, rank = get_grad(params, new_key, numsamples, N, model)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, e_loc, new_key, eigs, rank#, value




# numsamples = 100
# N = 20
# lr=5
# rng_key = jax.random.key(1)
# lmbda =3e-3


key = jax.random.key(42)
# N_LIST = [10,20,30,40,50, 100, 200]
# N = N_LIST[int(sys.argv[1])]
key1, key2 = jax.random.split(jax.random.key(1))
# x = jax.random.randint(key1, (5,N), 0, 2) # Dummy input data

rng_key = jax.random.key(1)
# lmbda =2e-3
# optimizer = optax.sgd(learning_rate=inverse_schedule)
# t1 = time.time()
# e_minsr, min_sr_vars, min_sr_time = minSR(N)
# t2 = time.time()
# min_sr_time = t2-t1
lr = 5.5e-3
optimizer = optax.adam(learning_rate=lr)
t1 = time.time()
e_adam, adam_vars, adam_time = ADAM(N)
t2 = time.time()
adam_time = t2-t1



# for n in N_LIST:
#   numsamples = 100
#   N = n
#   lr = 5
#   x = jax.random.randint(jax.random.key(2), (5,n), 0, 2) # Dummy input data
#   key, keyOld = jax.random.split(key)
#   params = model.init(key, x)
#   opt_state = optimizer.init(params)
#   numsamples = 100
#   N = n
#   lr = 5

#   t1 = time.time()
#   rng_key = jax.random.key(1)
#   energies = []
#   vars = []
#   print('Training started')
#   for i in range(400):
#   # for i in tqdm(range(1000), desc="Epochs"):
#       # params, opt_state, (_, eloc), rng_key = step(params, rng_key, opt_state)
#       params, opt_state, eloc, rng_key = step(params, rng_key, opt_state)
#       # if i % 200 == 0:
#       #   print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
#       energies.append(jnp.mean(eloc))
#       vars.append(jnp.var(eloc))

#   t2 = time.time()
#   times.append((n,t2-t1))
#   min_sr_energies.append((n, energies))
#   min_sr_vars.append((n, vars))



# #   _,e2 = adam_vars[i]
# plt.plot(jnp.log(jnp.array(min_sr_vars)), label = f'minSR')
# plt.plot(jnp.log(jnp.array(adam_vars)), label = f'Adam')
# #   plt.plot(jnp.log(jnp.array(e2)), label = 'Adam')
# plt.legend()
# plt.title(f'MinSR and Adam Ground State Energy n={N}')
# plt.xlabel('Iterations')
# plt.ylabel('log(var)')
# plt.savefig(folder + f'/Plots/OC-{numsamples}.pdf')
# plt.show()
     

