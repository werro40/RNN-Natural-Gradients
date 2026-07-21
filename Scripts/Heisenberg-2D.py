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

