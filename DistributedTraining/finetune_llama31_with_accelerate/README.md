# Finetune Llama3.1 70B with Accelerate

To run this example with FSDP, you need more than 2 GPUs, and each GPU should have at least 24GB memory.
It can be consumer GPUs such as the RTX 3090 or RTX 4090.

## Configuration

We need Hugging Face’s Accelerate. Make sure it’s installed and up to date:
```bash
pip install accelerate --upgrade
```

Then, configure it by running:
```bash
accelerate config
```

It will ask you several questions.
The goal here is to generate a configuration file that will be used for fine-tuning with FSDP.
Some of the questions can be difficult to answer if you don’t know well how FSDP works.
If this is the case, you can skip this step and use an existing configuration file, such as this one:
```
compute_environment: LOCAL_MACHINE                                                                                                                                           
debug: false                                                                                                                                                                 
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: true
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

You only have to modify `num_processes` which is the number of GPUs you have on your machine.
Then, save it into a file, e.g., config_fsdp.yaml.

## Fine-tuning

Use accelerate to launch the fine-tuning script:
```bash
accelerate launch --config_file config_fsdp.yaml fine_tuning_FSDP_QLoRA.py
```

### Trick 1

Since the model is split during fine-tuning, the checkpoints only contain pieces of the model.

To save the model, we need to gather all the pieces of the model on one device.
This is achieved by the following code that we have to run after the training (this code handles the (Q)LoRA case):
```python
fsdp_plugin = trainer.accelerator.state.fsdp_plugin
fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
        
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.
    set_state_dict_type("FULL_STATE_DICT")

```

### Trick 2

For QLoRA training, we need to prepare the model for training.
For single-GPU QLoRA fine-tuning, we would simply add this line:
```python
model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant':True})
```

It does the following:
    1) Cast the layernorm and the language modeling head in fp32
    2) Freeze the parameters of the models
    3) Make output embedding layer requires grads
    4) Activate gradient checkpointing

With FSDP, (1) doesn’t seem to be possible and triggers an error when the fine-tuning starts.
To avoid this casting, I implemented what `prepare_model_for_kbit_training` does, minus this first step:
```python
for name, param in model.named_parameters():
    # freeze base model's layers
    param.requires_grad = False

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
```
