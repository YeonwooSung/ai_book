"""
This code is a copied version of @simjeg's code from:
<https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag/comments>

Credits to @simjeg for this code.

The main aim of this code is to shard the Platypus2 70B model checkpoint and save it using safetensors.
"""

import gc
import json
from tqdm import tqdm
from pathlib import Path

import torch
from safetensors.torch import save_file


def save_layers(checkpoint_path):
    """
    Save the all layers of a model sharded checkpoint using safetensors.
    """

    checkpoint_path = Path(checkpoint_path)

    with open(checkpoint_path / 'pytorch_model.bin.index.json', 'rb') as f:
        index = json.load(f)['weight_map']

    n_layers = len(set([int(k.split('.')[2]) for k in index.keys() if 'model.layers' in k]))
    layers = ['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)] + ['model.norm.', 'lm_head.']
    shard = 0
    n_shards = len(set(index.values()))
    state_dict = {}

    for layer in tqdm(layers):

        # Optionnally load next shard
        shards = [int(v.split('-')[1]) for k, v in index.items() if k.startswith(layer)]
        if max(shards) > shard:
            shard += 1
            print(f'Loading shard {shard}/{n_shards}')
            state_dict.update(torch.load(checkpoint_path / f'pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin', map_location='cpu'))

        # Get layer state dict
        layer_state_dict = dict([(k, v) for k, v in state_dict.items() if k.startswith(layer)])

        # Save layer state dict as using safetensors
        save_file(layer_state_dict, checkpoint_path / (layer + 'safetensors'))

        # Free memory
        for k in layer_state_dict.keys():
            del state_dict[k]
        del layer_state_dict
        gc.collect()
