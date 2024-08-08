#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install --upgrade kagglehub -q\n!pip install ipywidgets -q\n!pip install tensorflow-cpu -q\n!pip install tensorflow_datasets -q\n!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q\n!pip install git+https://github.com/felafax/gemma.git -q\n!pip install qax -q\n!pip install jax-lorax -q\n')


# In[3]:


import os
os.environ['HF_HUB_CACHE'] = '/mnt/persistent-disk/hf/'
os.environ['HF_HOME'] = '/mnt/persistent-disk/hf/'
get_ipython().system('export HF_HUB_CACHE="/mnt/persistent-disk/hf/"')
get_ipython().system('export HF_HOME="/mnt/persistent-disk/hf/"')


# In[7]:


# @title Python imports

import enum
import re
import string
import pdb

# We import JAX and some related packages.
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial

# For LoRA
import lorax

# We will use HuggingFace's dataset, tokenizer, and model classes.
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator
from datasets import Dataset, load_dataset, concatenate_datasets
import torch

# Finally, we import Gemma.
from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib
import sentencepiece as spm


# In[5]:


jax.devices()


# ## Fine tuning the Gemma model

# In[180]:


import flax
from flax.traverse_util import flatten_dict
from flax.core.meta import unbox


def print_params(params):
    flat_params = flatten_dict(params)    
    for path, param in flat_params.items():
        # Join the path components to create a string name
        name = "/".join(str(x) for x in path)
        print(f"Name: {name}")
        # print(f"Shape: {param.shape}")
        # print(f"dtype: {param.dtype}")
        # print(f"Value: {param}")
        if isinstance(param, flax.core.meta.Partitioned):
            array = unbox(param)
        else:
            array = param
        print(jax.debug.visualize_array_sharding(array))
        print("-" * 40)


# ## Try LoRA with simpleNN

# In[145]:





# In[127]:


input_dim = 32 
hidden_dim = 8
output_dim = 1


# In[128]:


# Helper function for creating NamedSharding
def create_sharding(pspec):
    return NamedSharding(mesh, pspec)


# In[129]:


def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
  return NamedSharding(mesh, pspec)


# In[163]:


import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from flax import linen as nn
from flax.training import train_state
import optax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
import functools
from functools import partial
import jax
import jax.numpy as jnp

class SimpleNN(nn.Module):
    hidden_dim: int
    output_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = with_sharding_constraint(x, mesh_sharding(PartitionSpec('data', 'model')))

        x = nn.Dense(features=self.hidden_dim, 
                     kernel_init=nn.with_partitioning(nn.initializers.xavier_normal(), ('data', 'model')),
                     use_bias=False)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim, 
                     use_bias=False)(x)
        return x

# Set up the device mesh
devices = jax.devices()
device_mesh = mesh_utils.create_device_mesh((1, 4))
mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))

print(mesh)


# In[164]:


sample_batch = jnp.ones(shape=(1, input_dim))
x_sharding = mesh_sharding(PartitionSpec('data', 'model')) # dimensions: (batch, length)
sample_batch = jax.device_put(sample_batch, x_sharding)
jax.debug.visualize_array_sharding(x)


# In[165]:


# Initialize function
def init_fn(key, x, model, optimizer):
    params = model.init(key, x)  # Initialize the model
    state = train_state.TrainState.create(  # Create a `TrainState`
        apply_fn=model.apply,
        params=params['params'],
        tx=optimizer)
    return state


# In[166]:


model = SimpleNN(hidden_dim, output_dim)
optimizer = optax.adam(learning_rate=0.001)


# In[167]:


abstract_variables = jax.eval_shape(
    functools.partial(
        init_fn, 
        model=model, 
        optimizer=optimizer
    ),
    jax.random.PRNGKey(99),
    sample_batch
)


# In[135]:


abstract_variables


# In[168]:


state_sharding = nn.get_sharding(abstract_variables, mesh)


# In[138]:


state_sharding


# In[169]:


jit_init_fn = jax.jit(init_fn, 
                          static_argnums=(2, 3),
                          in_shardings=(mesh_sharding(pspec=()), x_sharding),  # PRNG key and x
                          out_shardings=state_sharding)


# In[170]:


initialized_state = jit_init_fn(jax.random.PRNGKey(99), sample_batch, model, optimizer)


# In[171]:


initialized_state.params['Dense_1']['kernel']


# In[181]:


print_params(initialized_state.params)


# In[182]:


def forward_pass(params, state, batch):
  input_images, labels = batch
    
  # call forward pass function.
  logits = state.apply_fn({"params": params}, input_images)

  # compute loss
  loss = optax.squared_error(logits, labels)
  loss = loss.mean()
  return loss, logits


# In[183]:


def backward_pass(state, batch):
  # create a function to compute gradients wrt to loss
  # returned by our `forward_pass` function.
  grad_fn = jax.value_and_grad(forward_pass, argnums=(0), has_aux=True)

  # compute gradients.
  (loss, _), grads = grad_fn(state.params, state, batch)

  # apply gradients.
  state = state.apply_gradients(grads=grads)

  return state


# In[207]:


@functools.partial(jax.jit, in_shardings=(state_sharding, x_sharding, None),
                   out_shardings=state_sharding)
def train_step(state, inputs, targets):
  return backward_pass(state, (inputs, targets))


# In[208]:


batch = (jnp.ones(shape=(1, input_dim)), jnp.zeros(shape=(1, output_dim)))


# In[209]:


batch0 = jax.device_put(batch[0], x_sharding)


# In[210]:


batch1 = jax.device_put(batch[1], mesh_sharding(PartitionSpec('data', None)))


# In[211]:


with mesh:
    new_state = train_step(initialized_state, batch0, batch1)


# In[212]:


new_state


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




