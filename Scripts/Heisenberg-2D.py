import sys
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Tuple, Union, Optional, Callable, Any
import optax
from functools import partial
from jax import jit
import time
import datetime, os
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import pickle
from Utils.utils import data_class, _flatten_jacobian, _unflatten_like_params, _apply_step, slurm_time_to_seconds
from utils.models import StackedPRNNModel as model
import math

jax.config.update("jax_enable_x64", True)
jax_dtype = jnp.float64

parser.add_argument("key", type=int, help="key")
parser.add_argument("-t", "--test", action="store_true", help="test run for low memory")
parser.add_argument("--time_limit", type=str, help="Time limit in slurm format HH:MM or D-HH:MM etc.", default="2-10:00:00")
parser.add_argument("--config", type=str, help="configuration file", required=True)


import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import linen as nn
from typing import List, Tuple, Union, Optional, Callable, Any
import optax
#from tqdm import tqdm



opt_dict = {  'title': f'noise',
                'opt': 'minsr',
                'trust_region': True,
                 'CN-1': False,
                 'CN-2': True,
                 'lr': 1e-1,
                 'm': 0.75,
                 'step_decay': 10000,
                 'min_lr': 2e-5,
                 'TIME_LIMIT': 5*3600,
                 'max_lambda': 1e9,
                 'lambda_inc': 1.1,
                 'lambda_dec': 0.8
                 }

lr = opt_dict['lr']
m = opt_dict['m']
step_decay = opt_dict['step_decay']
# TIME_LIMIT = opt_dict['TIME_LIMIT']
TIME_LIMIT = slurm_time_to_seconds(args.time_limit) - 2000*(1-int(args.test))
CN1_bool = int(opt_dict['CN-1'])
CN2_bool = int(opt_dict['CN-2'])
tr_bool = int(opt_dict['trust_region'])
max_lambda = opt_dict['max_lambda']
lambda_inc = opt_dict['lambda_inc']
lambda_dec = opt_dict['lambda_dec']
title=f'-ss-lm-tr-d-grid={args.grid}'
T1 = time.time()
# bool_tr = k%2
# bool_cn = int(k>1)
s = title+f''' 
Opt: {opt},
July 15 - Quadratic rho
July 17- Grid
lr = {lr}/(1+step/{step_decay})
m = 0.75
dh: {dh}, 
numsamples: {numsamples}
k={k}
reg: 1/var +  not CN**0.5
lr: {lr}
m: 0.75
lambda_reg = 2.13e-4
step_size: 10.5/var not with CN-2 = lambda_max
steps: {steps}
Not 10.5/tr rather 1/tr July 1st
max_lambda = {max_lambda}
grid: {args.grid}
 \n '''
# def matrix_init(key, shape, dtype=jax_dtype, normalization=1):
#     return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization

# with open('Experiments/TFIM/Results_minSR_experiment-2025-12-02/Model/model_params-200-minSR.pkl', 'r') as f:
#   params = pickle.load(f)
if args.test:
  folder = './Experiments/minSR/2DH-' + str(datetime.date.today()) + title + '-test'
else:
    folder = './Experiments/minSR/2DH-' + str(datetime.date.today()) + title
print(f'{folder}') #grep -rnw '/path/to/search' -e "[folder]"

os.makedirs(folder, exist_ok=True)
os.makedirs(folder+'/Data', exist_ok=True)
os.makedirs(folder+'/Plots', exist_ok=True)
os.makedirs(folder+'/Models', exist_ok=True)

# if not os.path.exists(folder + '/Checkpoints'):
#     os.mkdir(folder + '/Checkpoints')

with open(folder + '/docs.txt','w') as f:
    f.write(s)
        
def get_jac(params, samples):
    jacobian = jax.jacrev(log_probs_fun)(params, samples)
    return jacobian       

def eloc_final(params,numsamples_f = 3*10**4):
    key_f = jax.random.key(1)
    # numsamples_f = 10**4
    samples_f = model.apply(params,key_f,numsamples_f,Nx, Ny,method="sample")
    log_probs = model.apply(params,samples_f)
    e_loc = local_energy(samples_f, params, model, 0.5*log_probs)
    e_loc_mean = jnp.mean(e_loc)
    e_loc_err =jnp.sqrt( jnp.var(e_loc))/ jnp.sqrt(numsamples_f)
    E = (e_loc_mean, e_loc_err)
    return E
    
model = StackedRNNModel(d_hidden=dh, d_model=2, n_layers=1, RNNcell_type = "GRU")
#RNNcell_type can be "Vanilla", "GRU"
#LSTM still not working yet


key1, key2 = jax.random.split(jax.random.key(1))
x = jax.random.randint(key1, (2,4,4), 0, 2) # Dummy input data
# print(x)
params = model.init(key2, x) # Initialization call
k = int(sys.argv[1])
# pkl_path = "Experiments/Results_2DH_experiment-2025-11-09-Large-Scale/model_params-100000-minsr.pkl"
# with open(pkl_path, "rb") as f:
#     params = pickle.load(f)

def local_energy(samples, params, model, log_psi) -> List[float]:

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
        flipped_logpsi = 0.5*model.apply(params,flipped_state)
        output += (s[:, i,j] + s[:, i+1,j] == 1) *(-0.5)* jnp.exp(flipped_logpsi - log_psi)

        return s, output



    def step_fn_vertical(n, state):

        s, output = state
        _, Nx,Ny = s.shape

        j = (n//Nx) #set back to zero when equal to Nx-1
        i = n%Nx


        flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
        flipped_state = flipped_state.at[:, i,j+1].set(1 - flipped_state[:, i,j+1])
        flipped_logpsi = 0.5*model.apply(params,flipped_state)
        output += ((s[:, i,j] + s[:, i,j+1] == 1)*(-0.5))*jnp.exp(flipped_logpsi - log_psi)

        return s, output

    # Off Diagonal Term
    output = jnp.zeros((numsamples), dtype=jax_dtype)
    _, off_diag_term_vertical = jax.lax.fori_loop(0, Nx*(Ny-1), step_fn_vertical, (samples, output))
    _, off_diag_term_horizontal = jax.lax.fori_loop(0, (Nx-1)*(Ny), step_fn_horizontal, (samples, output))

    local_energies += off_diag_term_vertical +  off_diag_term_horizontal

    return local_energies
    
    
def log_probs_fun(params, samples):
    return 0.5*model.apply(params,samples)
    
def get_loss(params, key, numsamples, Nx, Ny, model):

    samples = model.apply(params,key, numsamples,Nx,Ny, method="sample")
    log_probs = model.apply(params,samples)

    e_loc = jax.lax.stop_gradient(local_energy(samples, params, model, 0.5*log_probs))
    e_avg = e_loc.mean()

    loss = jnp.mean(jnp.multiply(log_probs, e_loc) - jnp.multiply(e_avg, log_probs))
    return loss, e_loc

def make_training_step(model, optimizer, local_energy_fn, get_jac,
                        numsamples, Nx, Ny, batches, samples_per_batch, step_schedule):
    """Close over everything that never changes. Returns a pure
    (params, rng_key, opt_state, CN, lambda_reg) -> same signature function
    that can be jit-compiled cleanly."""

    def training_step(params, rng_key, opt_state, CN, lambda_reg, step):
        rng_key, new_key = jax.random.split(rng_key)

        # Sampling + local energies (was get_grad)
        samples = model.apply(params, new_key, numsamples, Nx, Ny, method="sample")
        log_probs = model.apply(params, samples)
        e_loc = local_energy_fn(samples, params, model, 0.5 * log_probs)
        e_loc_c = e_loc - e_loc.mean()

        # MinSR + LM update (was get_minSR_gradients)
        numsamples_ = samples.shape[0]
        samples_batched = samples.reshape(
            (batches, samples_per_batch, samples.shape[1], samples.shape[2])
        )
        jacobian_batched = jax.lax.map(lambda s: get_jac(params, s), samples_batched)
        jacobian = jax.tree.map(
            lambda x: x.reshape(numsamples_, *x.shape[2:]), jacobian_batched
        )

        jac, tree, shapes, slices = _flatten_jacobian(jacobian, numsamples_)
        jac0 = jac
        jac = jac - jnp.mean(jac, axis=0)
        jac = jac / jnp.sqrt(numsamples_)
        XdaggerX = jac @ jac.T
        D = jnp.diag(jnp.diag(XdaggerX)**(1/(args.key+1)))
        Id = jnp.eye(XdaggerX.shape[0])
        M = (1-args.grid) * D + (args.grid) * Id

        x = jax.scipy.linalg.solve(
            XdaggerX + lambda_reg * M,
            e_loc_c,
            assume_a="pos",
        )
        
        tau = e_loc_c-lambda_reg*x
        step_reg = 1/jnp.linalg.norm(tau)
        dtheta = step_reg * jac.T @ x
        grads = _unflatten_like_params(dtheta, tree, shapes, slices)

        # LM trust-region check using real re-evaluation (was lm_trial_step)
        trial_params = _apply_step(params, grads, step_schedule(step))
        trial_samples = model.apply(trial_params, new_key, numsamples, Nx, Ny, method="sample")
        trial_log_probs = model.apply(trial_params, trial_samples)
        e_loc_trial = local_energy_fn(trial_samples, trial_params, model, 0.5 * trial_log_probs)
        e_loc_trial_c = e_loc_trial - e_loc_trial.mean()

        # actual_reduction = 0.5 * (jnp.sum(e_loc_c**2) - jnp.sum(e_loc_trial_c**2))
        # predicted_reduction = 0.5 * (jnp.sum(e_loc_c**2) - step_schedule(step)**2 * step_reg**2 * lambda_reg**2 * jnp.sum(x**2))
        # predicted_reduction = 0.5 * (jnp.sum(e_loc_c**2) - jnp.linalg.norm(e_loc_c - step_schedule(step) * step_reg * tau)**2)
        ar2 = e_loc.mean() - e_loc_trial.mean()  # h(theta)-h(theta+delta_theta))
        pr2 = e_loc.mean() - 2*step_schedule(step) * e_loc.T @ jac0 @ dtheta +0.5 * step_schedule(step)**2 
        # jax.debug.print('actual reduction: {ar}, predicted reduction: {pr}', ar=actual_reduction, pr=predicted_reduction)
        # jax.debug.print('actual reduction2: {ar2}, predicted reduction2: {pr2}', ar2=ar2, pr2=pr2)
        # jax.debug.print('trust region: {eps}',eps = dtheta.T @ jac.T @ jac @ dtheta )
        # jax.debug.print('actual reduction: {ar}, predicted reduction: {pr}', ar=actual_reduction, pr=predicted_reduction)
        # rho = actual_reduction / (predicted_reduction + 1e-12)
        rho = ar2/(pr2 + 1e-12)
        # jax.debug.print('rho: {rho}, rho2: {rho2}', rho=rho, rho2=rho2)
        # rho=rho2

        lambda_reg_new = jnp.where(
            rho > 0.75, lambda_reg / lambda_inc,
            jnp.where(rho < 0.25, lambda_reg * lambda_inc, lambda_reg),
        )
        lambda_reg_new = jnp.clip(lambda_reg_new, 1e-8, max_lambda)

        # Optimizer update (was step_minSR)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)

        return params_new, opt_state_new, e_loc, new_key, CN, lambda_reg_new

    return jax.jit(training_step)




def train_minsr(params, keynum=0):
  train_step = make_training_step(
    model, optimizer, local_energy, get_jac,
    numsamples, Nx, Ny, batches, samples_per_batch, inverse_schedule
  )
  opt_state = optimizer.init(params)
  data = data_class(Energy=[],Var=[],Time=[],Lambda=[])
  CN=1
  print('Training started')
  rng_key = jax.random.key(keynum)
  t1 = time.time()
  lambda_reg = 1e-2
  for i in range(steps):
      print(i)
      # lambda_reg = 1e-3
      # params, opt_state, eloc, rng_key, CN, rank = step_minSR(params, rng_key, opt_state, CN, lambda_reg)
    #   params, opt_state, eloc, rng_key, CN,lambda_reg= step_minSR(params, rng_key, opt_state, CN, lambda_reg)
      params, opt_state, e_loc, rng_key, CN, lambda_reg = train_step(params, rng_key, opt_state, CN, lambda_reg,i)
      # params, opt_state, (_,eloc), rng_key = step_adam(params, rng_key, opt_state)

      update = [jnp.mean(e_loc),jnp.var(e_loc),time.time() - t1,lambda_reg]
      data.update(update)

      if math.isnan(data.Energy[-1]):
          print("NaN detected, stopping training.")
          break
    #   if i>3:
    #       if (data.Time[-1]-data.Time[1]) > TIME_LIMIT: #end after one hour of training
    #           print("Time limit exceeded, stopping training.")
    #           break
      if time.time() - T1 > TIME_LIMIT:
          print("Time limit exceeded, stopping training.")
          break
    #   params, opt_state, eloc, S_matrix, rng_key = step(params, rng_key, opt_state)
      if i % 5000 == 3:
          print(f't1: {data.Time[-1]:.4e}')
          e_loc_final_mean,e_loc_final_error=E_min_sr = eloc_final(params,numsamples_check)
          metadata = {'opt': opt, 'date':str(datetime.date.today()), 'folder': folder,'dh':dh,'Ns-train':numsamples, 'E/N': e_loc_final_mean/N**2, 'error/N': e_loc_final_error/N**2, 'Ns_check':numsamples_check}
          with open(folder + f'/Models/model_params-{opt}-s{i}-{k}.pkl', 'wb') as f:
            pickle.dump(params, f)
          data.plot("Var",folder=folder,log=True, label = f'{opt}, E/N:{e_loc_final_mean/N**2:.5f} +/- {e_loc_final_error/N**2:.3e}',i=args.key)
          data.plot("Lambda",folder=folder,log=True, label = f'{opt}, E/N:{e_loc_final_mean/N**2:.5f} +/- {e_loc_final_error/N**2:.3e}',i=args.key)
          data.save(folder=folder, metadata=metadata,i=args.key)
    
  return params, data #ranks, params

def inverse_schedule(step):
    return jnp.maximum(lr/(1+step/step_decay),lr/100)
optimizer = optax.sgd(learning_rate=inverse_schedule, momentum = m)
# optimizer = optax.adam(learning_rate=inverse_schedule)
# params = params0
opt_state = optimizer.init(params)

Nx = N
Ny = N
rng_key = jax.random.key(1)

#################
#.    train
##################
params, data = train_minsr(params)
  
##############
#    Save
##############
with open(folder + f'/Models/model_params-{steps}-{opt}-{k}.pkl', 'wb') as f:
  pickle.dump(params, f)

# eigenvals = jnp.array(eigenvals)
# log_res = jnp.array(log_res)
t2 = time.time()
e_loc_final_mean, e_loc_final_error = eloc_final(params,numsamples_final)

# jnp.save(folder + f"/Data/energies-{opt}-{steps}-{k}.npy", min_sr_energies)
# jnp.save(folder + f"/Data/vars-{opt}-{steps}-{k}.npy", min_sr_vars)
# jnp.save(folder + f"/Data/times-{opt}-{steps}-{k}.npy", times)
# jnp.save(folder + f"/Data/lambdas-{opt}-{steps}-{k}.npy", lambdas)
metadata = {'opt': opt, 'date':str(datetime.date.today()), 'folder': folder, 'E/N': e_loc_final_mean/N**2, 'error/N': e_loc_final_error/N**2, 'Ns_check':numsamples_final}
data.save(folder=folder, metadata=metadata,i=args.key)

data.plot("Var",folder=folder,log=True, label = f'{opt}, E/N:{e_loc_final_mean/N**2:.5f} +/- {e_loc_final_error/N**2:.3e}',i=args.key)
data.plot("Energy",folder=folder,log=False, label = f'{opt}, E/N:{e_loc_final_mean/N**2:.5f} +/- {e_loc_final_error/N**2:.3e}',i=args.key)
data.plot("Lambda",folder=folder,log=True, label = f'{opt}, E/N:{e_loc_final_mean/N**2:.5f} +/- {e_loc_final_error/N**2:.3e}',i=args.key)
# plt.plot(jnp.log(jnp.array(min_sr_vars)), label = f'{opt}, time: {min_sr_time:.4f}, Energy = {E_min_sr[0]:.6f} +/- {E_min_sr[1]:.2e}')
#plt.plot(jnp.log(jnp.array(adam_vars)), label = f'Adam, time: {t2-t1:.4f}, Energy = {E_adam[0]:.4f} +/- {E_adam[1]:.3e}')
# plt.plot(jnp.log(jnp.array(adam_vars)), label = f'Adam, time: {adam_time}')
##   plt.plot(jnp.log(jnp.array(e2)), label = 'Adam')
# plt.legend()
# plt.title(f'Minsr Heisenberg Vars 10x10')
# plt.xlabel('Iterations')
# plt.ylabel('log(var)')
# plt.savefig(folder + f'/Plots/minsr-Heis-{steps}-vars-{k}.pdf')
# plt.close()



# plt.plot(min_sr_energies, label = f'{opt}, E/N = {E_min_sr[0]/N**2:.6f} +/- {E_min_sr[1]/N**2:.2e}')
# plt.axline((0,E_min_sr[0]),(1,E_min_sr[0]), c='r', ls = '--')
#plt.plot(adam_energies, label = f'Adam, time: {t2-t1:.4f}, Energy = {E_adam[0]:.4f} +/- {E_adam[1]:.3e}')
#plt.axline((0,E_adam[0]),(1,E_adam[0]), c='g', ls = '--')
#plt.plot(jnp.log(jnp.array(adam_vars)), label = f'Adam, time: {adam_time}')
##   plt.plot(jnp.log(jnp.array(e2)), label = 'Adam')
# plt.legend()
# plt.title(f'MinSR Heisenberg E 10x10')
# plt.xlabel('Iterations')
# plt.ylabel('energies')
# plt.savefig(folder + f'/Plots/minsr-Heis-{steps}-energies-{k}.pdf')
# plt.close()

# plt.plot(jnp.array(lambdas), label = f'{opt}, E/N = {E_min_sr[0]/N**2:.6f} +/- {E_min_sr[1]/N**2:.2e}')
# #plt.plot(adam_energies, label = f'Adam, time: {t2-t1:.4f}, Energy = {E_adam[0]:.4f} +/- {E_adam[1]:.3e}')
# #plt.axline((0,E_adam[0]),(1,E_adam[0]), c='g', ls = '--')
# #plt.plot(jnp.log(jnp.array(adam_vars)), label = f'Adam, time: {adam_time}')
# ##   plt.plot(jnp.log(jnp.array(e2)), label = 'Adam')
# plt.legend()
# plt.title(r'MinSR Heisenberg 10x10; $\lambda = 10^{-3}\delta G$')
# plt.xlabel('Iterations')
# plt.ylabel(r'$\lambda$')
# plt.savefig(folder + f'/Plots/minsr-Heis-{steps}-lambdas-{k}.pdf')
# plt.close()

