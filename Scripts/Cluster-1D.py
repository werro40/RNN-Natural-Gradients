
# @title Cluster RNN Model
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from typing import List, Tuple, Union, Optional, Callable, Any
import optax
from jax import jit
jax.config.update("jax_enable_x64", True)
from tqdm import tqdm
import matplotlib.pyplot as plt
import optuna
import datetime
import sys, os, time
import pickle
k = int(sys.argv[1])

# N_array = [64, 75, 90,100]
# step_array = [5000, 5000, 7000, 10**4]
# lrs = [1e-2,9e-3,8e-3,7e-3,6e-3]
test = False
if test:
    N = 2
    steps = 5
    dh=1
    numsamples = 10
    final_samples = int(2e0)
else:
  N = 30
  numsamples = 100
  dh = 20
  steps= int(1e4) 
  final_samples = int(2e4)

lr = 1e-3


s = f'''Bismillahirahman alrahim ,
no e,N=100
Optimizer = minsr {lr:.3e}, 
dh={dh},
samples={numsamples} 
steps={steps}
steps = variable'''

folder = './Experiments/Cluster/Results-' + str(datetime.date.today()) +f'-dh-20'

if not os.path.exists(folder):
    os.mkdir(folder)
    os.mkdir(folder + '/Plots')
    os.mkdir(folder + '/Data')
    os.mkdir(folder + '/Model')

with open(folder + '/docs.txt','w') as f:
    f.write(s)
# with open(folder + '/docs.txt','w') as f:
#     f.write(s)
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
    
# @title Helpers
def local_energy(samples, params, model, log_psi) -> List[complex]:
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


def get_loss(params, key, NUMBER_OF_SAMPLES, N, model):
    samples = model.apply(params, key, NUMBER_OF_SAMPLES, N,
                          method="sample")
    log_psi = model.apply(params, samples)
    e_loc = jax.lax.stop_gradient(local_energy(samples, params, model, log_psi))
    e_avg = e_loc.mean()

    loss = 2*jnp.real(jnp.mean(jnp.conjugate(log_psi)*(e_loc-e_avg)))
    return loss, e_loc


def final_energy(params, key, model, N, num_samples_final):
  samples = model.apply(params,key, num_samples_final, N, method="sample")
  log_psi = model.apply(params,samples)
  e_loc = local_energy(samples, params, model, log_psi)
  return jnp.mean(e_loc), jnp.var(e_loc), jnp.sqrt(jnp.var(e_loc))/jnp.sqrt(num_samples_final)


def log_probs_fun_r(params, samples):
    return jnp.real(0.5*model.apply(params,samples))
def log_probs_fun_i(params, samples):
    return jnp.imag(0.5*model.apply(params,samples))
# param_count = sum(x.size for x in jax.tree_leaves(params))


def get_minSR_gradients(params, samples, local_energies,CN):
  jacobian_r = jax.jacrev(log_probs_fun_r)(params, samples)
  jacobian_i = jax.jacrev(log_probs_fun_i)(params, samples)

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
  XdaggerX = X@X.T

  # lambda_reg = jnp.var(jac)
  # lambda_reg = jnp.trace(XdaggerX)
  lambda_max = jnp.linalg.eigvalsh(XdaggerX)
  norm  = jnp.linalg.norm(XdaggerX)
  lambda_reg = 2e-3
  # cfac = jax.scipy.linalg.cho_factor(x)
  # y = jax.scipy.linalg.cho_solve(cfac, b)
  # print(lambda_reg)
#   XdaggerX_inv =  jax.scipy.linalg.inv((XdaggerX + lambda_reg * jnp.eye(X.shape[0])))
  # XdaggerX_inv = minSR_pseudo_inverse(XdaggerX, soft=True)
#   norm_inv  = jnp.linalg.norm(XdaggerX_inv)
  CN_t = jnp.log(norm)
#   k=0.5
  # step_reg =2**0.5/(CN_t/CN + CN/CN_t)**0.5 #1/( 1 + abs(CN-CN_t)**0.25)#10.5/jnp.linalg.norm(local_energies)*
  step_reg=1
  cfac = jax.scipy.linalg.cho_factor(XdaggerX + lambda_reg * jnp.eye(X.shape[0]))
  gradients =  step_reg * X.T @ jax.scipy.linalg.cho_solve(cfac, f)

  #gradients *= jnp.min(1,5/grad_norm)
#   CN = CN_t

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

  return original_grad, CN
  # return original_grad, S_matrix

def get_grad(params, key, numsamples, N, model, CN):
    samples = model.apply(params,key,numsamples,N,method="sample")
    log_amps = model.apply(params,samples)
    e_loc = local_energy(samples, params, model, log_amps)
    # e_loc_c = e_loc - e_loc.mean()
    grads, CN = get_minSR_gradients(params, samples, e_loc, CN)

    return grads, e_loc, CN
    # return grads, e_loc, S_matrix
    
@partial(jit)
def step(params, rng_key, opt_state, CN):
    rng_key, new_key = jax.random.split(rng_key)
    grads, e_loc, CN = get_grad(params, new_key, numsamples, N, model, CN)
    # grads, e_loc, S_matrix = get_grad(params, new_key, numsamples, Nx, Ny, model)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, e_loc, new_key, CN
    # return params, opt_state, e_loc, S_matrix, new_key
# @partial(jit, static_argnums=(3,))
# def step_adam(params, rng_key, opt_state, get_loss=get_loss):
#     rng_key, new_key = jax.random.split(rng_key)
#     value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, numsamples,N, model)
#     # grads_flat, _ = jax.tree_util.tree_flatten(grads)
#     # global_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
#     # g_factor = jnp.minimum(1.0, grad_clip_norm / global_l2)
#     # grads = jax.tree_util.tree_map(lambda g: g * g_factor, grads)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, value, new_key
model = CRNNModel(output_dim=2, num_hidden_units=dh, RNNcell_type='GRU')

key1, key2 = jax.random.split(jax.random.key(1))
x = jax.random.randint(key1, (5, 10), 0, 2)  # Dummy input data
params = model.init(key2, x)



def inverse_schedule(step):
    return lr/(1+step/5000)
# optimizer = optax.sgd(learning_rate=9.9e-3, momentum = 0.5)
optimizer = optax.sgd(learning_rate=inverse_schedule, momentum = 0.1)

opt_state = optimizer.init(params)
CN = 1
rng_key = jax.random.key(1)
print('Training started')
energies = []
vars = []
times = []
# optimizer = optax.sgd(learning_rate=lr, momentum = m)
# optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(params)
rng_key = jax.random.key(1)
@partial(jit)
def step(params, rng_key, opt_state, CN):
    rng_key, new_key = jax.random.split(rng_key)
    grads, e_loc, CN = get_grad(params, new_key, numsamples, N, model, CN)
    # grads, e_loc, S_matrix = get_grad(params, new_key, numsamples, Nx, Ny, model)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, e_loc, new_key, CN

# @partial(jit, static_argnums=(3,))
# def step_adam(params, rng_key, opt_state, get_loss=get_loss):
#     rng_key, new_key = jax.random.split(rng_key)
#     value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, numsamples,N, model)
#     # grads_flat, _ = jax.tree_util.tree_flatten(grads)
#     # global_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads_flat]))
#     # g_factor = jnp.minimum(1.0, grad_clip_norm / global_l2)
#     # grads = jax.tree_util.tree_map(lambda g: g * g_factor, grads)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, value, new_key
t_start = time.time()
for i in range(steps):
    # params, opt_state, (_,eloc), rng_key = step_adam(params, rng_key, opt_state)
    params, opt_state, eloc, rng_key, CN = step(params, rng_key, opt_state, CN)
    t = time.time()-t_start
    times.append(t)
    # params, opt_state, eloc, S_matrix, rng_key = step(params, rng_key, opt_state)
    if i % 5000 == 200:
      print("Step = ",i, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
      jnp.save(folder+f'/Data/Cluster-{N}-energies-{i}.npy',jnp.array(energies))
      jnp.save(folder+f'/Data/Cluster-{N}-vars-{i}.npy',jnp.array(vars))
      jnp.save(folder+f'/Data/Cluster-{N}-times-{i}.npy',jnp.array(times))

    energies.append(jnp.mean(eloc))
    vars.append(jnp.var(jnp.real(eloc)))
    # grads.append(stats["grad"])
    # li.append(stats["norm_inv"])
with open(folder + f'/Model/model_params-{N}-minsr.pkl', 'wb') as f:
  pickle.dump(params, f)
# pbar.set_postfix(loss = f'{jnp.log(vars[-1]):.4f},{energies[-1]:.3f}')
# vars_array.append((N_reg,jnp.log(jnp.array(vars))))
# vars_minsr = vars

samples = model.apply(params,key1,final_samples,N,method="sample")
log_amps = model.apply(params,samples)
e_loc = local_energy(samples, params, model, log_amps)
E_min_sr = (jnp.real(jnp.mean(e_loc)),jnp.sqrt(jnp.var(abs(e_loc))/10**4))

jnp.save(folder+f'/Data/Cluster-{N}-energies.npy',jnp.array(energies))
jnp.save(folder+f'/Data/Cluster-{N}-vars.npy',jnp.array(vars))
jnp.save(folder+f'/Data/Cluster-{N}-times.npy',jnp.array(times))

plt.plot(jnp.array(energies), label = f'minsr, E:{E_min_sr[0]:.5f} +/- {E_min_sr[1]:.3e}')
plt.axline((0,E_min_sr[0]),(1,E_min_sr[0]), c='r', ls = '--')
plt.legend()
plt.title(f'Cluster State N={N}, dh={dh}, samples={numsamples}')
plt.savefig(folder + f'/Plots/Cluster-{N}-{steps}-dh.pdf')
plt.close()

# jnp.save(jnp.array(vars),f'vars-{N}-{steps}.npy')
# def objective_minsr(trial):
#   lr = trial.suggest_float('lr', 1e-4, 0.1, log=True)
#   m = trial.suggest_float('m', 0, 0.8)
#   energy = train(lr,m)
#   return energy

# study = optuna.create_study(direction = "minimize")
# study.optimize(objective_minsr, n_trials = 100)
# study.best_params

# with open(folder+f"/optuna_all_trials-{N}.txt", "w") as f:
#     for trial in study.trials:
#         f.write(f"Trial {trial.number}\n")
#         f.write(f"  Value: {trial.value}\n")
#         f.write("  Params:\n")
#         for k, v in trial.params.items():
#             f.write(f"    {k}: {v}\n")
#         f.write("\n")
#         f.write(f"best params: {study.best_params}")
#         f.write(f"Studying N=10, cluster lr and m no decay")