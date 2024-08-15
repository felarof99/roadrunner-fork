#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning Gemma2 2B model on Roadrunner with JAX, Flax.
# 
# We have adopted the Gemma notebook from Google Deepmind to use HuggingFace's libraries, added support for doing **model parallel training** and simplified the setup.

# ## Setup 

# In[1]:


import os
import sys
import importlib
def import_local_module(module_path: str):
    sys.path.append('')
    module = importlib.import_module(module_path)
    return importlib.reload(module)

# Import felafax libraries
setup = import_local_module("trainer_engine.setup")
setup.setup_environment()


# In[2]:


# get_ipython().run_cell_magic('capture', '', '!pip install --upgrade kagglehub -q\n!pip install ipywidgets -q\n!pip install torch --index-url https://download.pytorch.org/whl/cpu -q\n!pip install git+https://github.com/felafax/gemma.git -q\n!pip install qax -q\n!pip install jax-lorax -q\n')


# In[3]:


globals().update(setup.setup_imports())

utils = import_local_module("trainer_engine.utils")
training_pipeline = import_local_module("trainer_engine.training_pipeline")


# ## Step 0: Input your HF username, token and download model weights

# In[4]:


# HuggingFace username and token to use when downloading.
MODEL_NAME="felafax/gemma-2-2b-it-JAX"
HUGGINGFACE_USERNAME = input("INPUT: Please provide your HUGGINGFACE_USERNAME: ")
HUGGINGFACE_TOKEN = input("INPUT: Please provide your HUGGINGFACE_TOKEN: ")

model_name=MODEL_NAME
hugging_face_token=HUGGINGFACE_TOKEN


# In[14]:


from huggingface_hub import snapshot_download
# Download the model to disk.
ckpt_path = snapshot_download(repo_id=MODEL_NAME, token=HUGGINGFACE_TOKEN)
vocab_path = os.path.join(ckpt_path, 'tokenizer.model')


# In[15]:


# Load the downloaded model.
params = {"params": params_lib.load_and_format_params(os.path.join(ckpt_path, 'gemma2-2b-it'))['transformer']}
model_config = transformer_lib.TransformerConfig.gemma2_2b(cache_size=30)
model = transformer_lib.Transformer(config=model_config)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    token=HUGGINGFACE_TOKEN
)


# ## Step 1: prepare the dataset
# 
# For this project, we're utilizing the refined **Alpaca dataset**, curated by yahma. This dataset is a carefully filtered selection of 52,000 entries from the original Alpaca collection. Feel free to substitute this section with your own data preparation code if you prefer.
# 
# It's crucial to include the EOS_TOKEN (End of Sequence Token) in your tokenized output. Failing to do so may result in endless generation loops.

# In[16]:


def get_dataset(*, tokenizer, batch_size=1, max_length=32, max_examples=None):
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

    def _custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
        """
        Collates batch items and converts PyTorch tensors to JAX arrays.
        Applies default_data_collator, then converts tensors to JAX format.
        """
        collated = default_data_collator(batch)
        jax_batch = {}
        for key, value in collated.items():
            jax_batch[key] = jnp.array(value.numpy()) if isinstance(value, torch.Tensor) else value
        
        return jax_batch

    # Load and preprocess the dataset
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    if max_examples:
        dataset = dataset.select(range(max_examples))
    dataset = dataset.map(_format_prompts, batched=True)

    # Create train and test dataset.
    ds = dataset.train_test_split(test_size=0.15)
    for split in ['train', 'test']:
        ds[split] = ds[split].map(_tokenize, batched=True, remove_columns=dataset.column_names)

    # Create DataLoaders
    dataloader_args = dict(shuffle=True, batch_size=batch_size, collate_fn=_custom_collate_fn)
    train_dataloader = torch.utils.data.DataLoader(ds['train'], **dataloader_args)
    test_dataloader = torch.utils.data.DataLoader(ds['test'], **dataloader_args)

    return train_dataloader, test_dataloader


# In[17]:


# # Uncomment to test dataset pipeline
# def test_dataset_pipeline(tokenizer):
#     """Print shapes of first batch to verify dataset pipeline."""
#     train_loader, _ = get_dataset(tokenizer=tokenizer, batch_size=1, max_length=512)
#     batch = next(iter(train_loader))
#     print("Input tokens shape:", batch['input_tokens'].shape)
#     print("Target mask shape:", batch['target_mask'].shape)
# test_dataset_pipeline(tokenizer)


# ## Step 2: Train the model by configuring the hyperparameters below.

# In[18]:


@chex.dataclass(frozen=True)
class TrainingConfig:
  learning_rate: float
  num_epochs: int
  max_steps: int | None = None  # Maximum number of training steps (if None, train for full num_epochs)

  # Dataset config
  batch_size: int = 512
  max_length: int = 512  # Max sequence length during fine-tuning
  dataset_size_limit: int | None = None    # Limit on number of dataset rows for faster training (if None, use full dataset)

  # Misc config
  print_every_n_steps: int = 1


# In[19]:


training_cfg = TrainingConfig(learning_rate=1e-5,
                              num_epochs=1,
                              max_steps=40, 
                              dataset_size_limit=None)


# In[20]:


train_dataloader, val_dataloader = get_dataset(tokenizer=tokenizer, max_length=training_cfg.max_length, max_examples=training_cfg.dataset_size_limit)
optimizer = optax.sgd(training_cfg.learning_rate)


# In[21]:


# Set up the device mesh for distributing the model across TPU cores and do model parallel training.
devices = jax.devices()
device_mesh = mesh_utils.create_device_mesh((1, 4, 1))
mesh = Mesh(devices=device_mesh, axis_names=('data', 'model', 'replica'))


# **NOTE**: It's important to note that the **time to first step of training will be slow**. This is because XLA takes time initially to compile the computational graph. However, once the compilation is complete, subsequent steps will run much faster using compiled+cached graph, and leveraging the full power of the all TPU cores for accelerated training.

# In[22]:


state = training_pipeline.train_loop(model=model,
                    tokenizer=tokenizer,
                    params=params,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    training_cfg=training_cfg, 
                    mesh = mesh)

