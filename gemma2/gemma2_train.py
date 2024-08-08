#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning Gemma2 2B model on Roadrunner with JAX, Flax.
# 
# We have adopted the Gemma2 notebook from Google Deepmind to use HuggingFace's libraries and and simplified the steps.

# ## Setup

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install --upgrade kagglehub -q\n!pip install ipywidgets -q\n!pip install torch --index-url https://download.pytorch.org/whl/cpu -q\n!pip install git+https://github.com/felafax/gemma.git -q\n!pip install qax -q\n!pip install jax-lorax -q\n')


# In[2]:


import os
os.environ['HF_HUB_CACHE'] = '/mnt/persistent-disk/hf/'
os.environ['HF_HOME'] = '/mnt/persistent-disk/hf/'
get_ipython().system('export HF_HUB_CACHE="/mnt/persistent-disk/hf/"')
get_ipython().system('export HF_HOME="/mnt/persistent-disk/hf/"')


# In[93]:


# @title Python imports

import enum
import re
import string
from dataclasses import dataclass

# We import JAX and some related packages.
import chex
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
from flax.traverse_util import flatten_dict
from flax.core.meta import unbox

import optax
import functools
from functools import partial

# Model partitioning related imports
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils


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


# In[4]:


# HuggingFace username and token to use when downloading.
MODEL_NAME="felafax/gemma-2-2b-it-Flax"
HUGGINGFACE_USERNAME = input("INPUT: Please provide your HUGGINGFACE_USERNAME: ")
HUGGINGFACE_TOKEN = input("INPUT: Please provide your HUGGINGFACE_TOKEN: ")

model_name=MODEL_NAME
hugging_face_token=HUGGINGFACE_TOKEN


# In[6]:


get_ipython().run_cell_magic('capture', '', "from huggingface_hub import snapshot_download\n\nckpt_path = snapshot_download(repo_id=MODEL_NAME, token=HUGGINGFACE_TOKEN)\nvocab_path = os.path.join(ckpt_path, 'tokenizer.model')\n")


# ## Fine tuning the Gemma model

# In[13]:


import flax
from flax.traverse_util import flatten_dict

def print_params(params):
    flat_params = flatten_dict(params)    
    for key, value in flat_params.items():
        name = "/".join(str(x) for x in key)
        print(f"Name: {name}")
        # print(f"Shape: {value.shape}")
        # print(f"dtype: {value.dtype}")
        # print(f"Value: {value}")
        if isinstance(value, flax.core.meta.Partitioned):
            array = unbox(value)
        else:
            array = value
        print(jax.debug.visualize_array_sharding(array))
        print("-" * 40)


# ## Step 1: prepare the dataset
# 
# For this project, we're utilizing the refined **Alpaca dataset**, curated by yahma. This dataset is a carefully filtered selection of 52,000 entries from the original Alpaca collection. Feel free to substitute this section with your own data preparation code if you prefer.
# 
# It's crucial to include the EOS_TOKEN (End of Sequence Token) in your tokenized output. Failing to do so may result in endless generation loops.

# In[24]:


def get_dataset(*, tokenizer, batch_size=1, max_length=32, max_examples=32):
    # Define Alpaca prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction: {}
    
    ### Input: {}
    
    ### Response: {}"""
    
    EOS_TOKEN = tokenizer.eos_token
    
    # Define formatting function.
    def _format_prompts(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    def _tokenize(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length+1)
        tokenized['input_ids'] = [input_id[:-1] for input_id in tokenized['input_ids']]
        tokenized['target_mask'] = [input_id[:-1] for input_id in tokenized['attention_mask']]
        return {
            'input_tokens': tokenized['input_ids'],
            'target_mask': tokenized['target_mask']
        }

    def _custom_collate_fn(batch):
        """Applies default_collate_fn from transformers and converts to JAX NumPy arrays."""
        batch = default_data_collator(batch)
        jax_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                jax_batch[key] = jnp.array(value.numpy())
            else:
                jax_batch[key] = value
        
        return jax_batch

    # Load and preprocess the dataset.
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    if max_examples:
        dataset = dataset.select(range(max_examples))
    dataset = dataset.map(_format_prompts, batched=True)

    # Create train and test dataset.
    ds = dataset.train_test_split(test_size=0.15)
    ds['train'] = ds['train'].map(_tokenize, batched=True, remove_columns=dataset.column_names)
    ds['test'] = ds['test'].map(_tokenize, batched=True, remove_columns=dataset.column_names)

    # Create DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        ds['train'],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=_custom_collate_fn
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        ds['test'],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=_custom_collate_fn
    )

    return train_dataloader, test_dataloader


# In[25]:


# # # Test Dataset
train_dataloader, _ = get_dataset(tokenizer=tokenizer)
for i, batch in enumerate(train_dataloader):
    if i>10:
        break
    input_ids, attention_mask = (
        batch["input_tokens"],
        batch["target_mask"],
        
    )
    print(input_ids.shape)
    print()
    print(attention_mask.shape)


# In[69]:


def forward_and_loss_fn(params,
                        *,
                        state,
                        input_tokens: jax.Array,            # Shape [B, L]
                        input_mask: jax.Array,              # Shape [B, L]
                        positions: jax.Array,               # Shape [B, L]
                        attention_mask: jax.Array,          # [B, L, L]
                        ) -> jax.Array:
  """Forward pass and loss function.

  Args:
    params: model's input parameters.
    model: gemma transformer model to call.
    input_tokens: input tokens sequence, shape [B, L].
    input_mask: tokens to ignore when computing the loss, shape [B, L].
    positions: relative position of each token, shape [B, L].
    attention_mask: input attention mask, shape [B, L].

  Returns:
    Softmax cross-entropy loss for the next-token prediction task.
  """

  # Forward pass on the input data.
  # No attention cache is needed here.
  logits, _ = state.apply_fn(
        params,
        input_tokens,
        positions,
        None,              # Attention cache is None.
        attention_mask,
    )

  # Exclude the last step as it does not appear in the targets.
  logits = logits[:, :-1]

  # Similarly, the first token cannot be predicteds.
  target_tokens = input_tokens[:, 1:]
  target_mask = input_mask[:, 1:]

  # Convert the target labels into one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

  # Normalisation factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the nll loss.
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor


# The Gemma transformer requires an attention mask and position vector alongside each input. We can conveniently generate these using the following function:

# In[70]:


def get_attention_mask_and_positions(example: jax.Array,
                                     pad_id : int,
                                     )-> tuple[jax.Array, jax.Array]:
  """Builds the position and attention mask vectors from the given tokens."""
  pad_mask = example != pad_id
  current_token_position = transformer_lib.build_positions_from_mask(pad_mask)
  attention_mask = transformer_lib.make_causal_attn_mask(pad_mask)
  return current_token_position, attention_mask


# We can now build the train_step function which performs the backward pass and updates the model's parameters accordingly.

# In[72]:


def train_step(state,
               params,
               optimizer: optax.GradientTransformation,
               opt_state: optax.OptState,
               pad_id: int,
               batch):
  """Train step.

  Args:
    model: gemma transformer model.
    params: model's input parameters.
    optimizer: optax optimizer to use.
    opt_state: input optimizer's state.
    pad_id: id of the pad token.
    batch: input batch.

  Returns:
    Training loss, updated parameters, updated optimizer state.
  """
  # Build the position and attention mask vectors.
  positions, attention_mask = get_attention_mask_and_positions(batch['input_tokens'], pad_id)

  # Forward and backward passes
  train_loss, grads = jax.value_and_grad(forward_and_loss_fn)(params,
                                                             state=state,
                                                             input_tokens=batch['input_tokens'],
                                                             input_mask=batch['target_mask'],
                                                             positions=positions,
                                                             attention_mask=attention_mask)
  # Update the parameters
  # updates, opt_state = optimizer.update(grads, opt_state)
  # params = optax.apply_updates(params, updates)
  state = state.apply_gradients(grads=grads)

  return train_loss, params, opt_state


# Similarly, we build a `validation_step` function without backward pass.

# And now the training loop itself.

# In[8]:


jax.devices()


# In[11]:


# Set up the device mesh
devices = jax.devices()
device_mesh = mesh_utils.create_device_mesh((1, 4))
mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))


# In[111]:


# Load parameters.
params = {"params": params_lib.load_and_format_params(os.path.join(ckpt_path, 'gemma2-2b-it'))}


# In[113]:


# Load model config.
config = transformer_lib.TransformerConfig.gemma2_2b(cache_size=30)
model = transformer_lib.Transformer(config=config)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    token=HUGGINGFACE_TOKEN
)
optimizer = optax.sgd(training_cfg.learning_rate)


# In[112]:


def init_fn(params, model, optimizer):
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=optimizer)
    return state


# In[108]:


def init_fn(params, model, optimizer, mesh):
    def shard_param(param):
        if len(param.shape) == 1:
            return NamedSharding(mesh, P('model'))
        elif len(param.shape) == 2:
            return NamedSharding(mesh, P('data', 'model'))
        else:
            return NamedSharding(mesh, P(None))

    sharded_params = jax.tree_util.tree_map(shard_param, params['params'])
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=sharded_params,
        tx=optimizer)
    return state


# In[116]:


abstract_variables = jax.eval_shape(
    functools.partial(
        init_fn,
        params=params,
        model=model, 
        optimizer=optimizer,
        # mesh=mesh
    ),
)
abstract_sharded_state = nn.get_sharding(abstract_variables, mesh)


# In[117]:


def shard_params_pytree(params, mesh):
    def shard_param(param):
        # Define sharding based on param shape
        if len(param.shape) == 1:
            # For 1D tensors (e.g., biases), shard across 'model' dimension
            return NamedSharding(mesh, P('model'))
        elif len(param.shape) == 2:
            # For 2D tensors (e.g., weight matrices), shard across both dimensions
            return NamedSharding(mesh, P('data', 'model'))
        else:
            # For higher-dimensional tensors, you might need a more complex strategy
            return NamedSharding(mesh, P(None))  # Replicate by default

    return jax.tree_util.tree_map(shard_param, params)


# In[118]:


params = shard_params_pytree(params, mesh)


# In[ ]:





# In[123]:


init_fn_jitted = jax.jit(init_fn, 
                         static_argnums=(1, 2),
                         out_shardings=abstract_sharded_state)
                         


# In[124]:


with mesh:
    sharded_state = init_fn_jitted(params, model, optimizer)


# In[39]:


sample_batch = next(iter(train_dataloader))


# In[80]:


sample_batch = jax.device_put(sample_batch, NamedSharding(mesh, PartitionSpec('data', 'model')))


# In[ ]:





# In[ ]:


compiled_train_step = jax.jit(train_step, static_argnames=['model', 'optimizer'])
opt_state = optimizer.init(params)


# In[ ]:


train_loss, params, opt_state = train_step(model=model,
                                            params=params,
                                            optimizer=optimizer,
                                            opt_state=opt_state,
                                            pad_id=tokenizer.pad_token_id,
                                            example=train_example)


# In[ ]:


training_cfg = TrainingConfig(learning_rate=1e-4,
                              num_epochs=1,
                              eval_every_n=20,
                              batch_size=1,
                              max_steps=10)

params = train_loop(model=model_2b,
                    params={'params': params['transformer']},
                    train_dataloader=train_dataloader,
                    tokenizer=tokenizer,
                    training_cfg=training_cfg)


# In[34]:


@chex.dataclass(frozen=True)
class TrainingConfig:
  learning_rate: float
  num_epochs: int
  eval_every_n: int
  batch_size: int
  max_steps: int | None = None

from dataclasses import dataclass
import numpy as np


def train_loop(
    model: transformer_lib.Transformer,
    params,
    train_dataloader,
    tokenizer,
    training_cfg: TrainingConfig):


  compiled_train_step = jax.jit(train_step, static_argnames=['model', 'optimizer'])
  optimizer = optax.sgd(training_cfg.learning_rate)
  opt_state = optimizer.init(params)

  n_steps = 0
  avg_loss=0

  for i, train_example in enumerate(train_dataloader):
    train_loss, params, opt_state = train_step(model=model,
                                                        params=params,
                                                        optimizer=optimizer,
                                                        opt_state=opt_state,
                                                        pad_id=tokenizer.pad_token_id,
                                                        example=train_example)
    n_steps += 1
    avg_loss += train_loss
    print(f"train_loss {train_loss}")
    if training_cfg.max_steps is not None and n_steps > training_cfg.max_steps:
      break
  return params


# We can fine-tune our model on a limited number of steps.

# In[35]:





# In[33]:




