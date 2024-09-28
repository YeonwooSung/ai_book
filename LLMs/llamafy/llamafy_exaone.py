import torch
import gc

from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def unload_model(model):
    """Clear memory by deleting a model and calling the garbage collector."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_llama_config(exaone_config):
    """Create and return a Llama configuration based on EXAONE config."""
    return LlamaConfig(
        vocab_size=exaone_config.vocab_size,
        hidden_size=exaone_config.hidden_size,
        intermediate_size=exaone_config.intermediate_size,
        num_hidden_layers=exaone_config.num_layers,
        num_attention_heads=exaone_config.num_attention_heads,
        max_position_embeddings=exaone_config.max_position_embeddings,
        rms_norm_eps=exaone_config.layer_norm_epsilon,
        num_key_value_heads=exaone_config.num_key_value_heads,
        rope_theta=exaone_config.rope_theta,
        bos_token_id=exaone_config.bos_token_id,
        eos_token_id=exaone_config.eos_token_id,
        pad_token_id=exaone_config.pad_token_id,
        attention_bias=False,
    )

def copy_embedding_weights(llama_model, exaone_model):
    """Copy embedding weights from EXAONE to Llama model."""
    llama_model.model.embed_tokens.weight.data = exaone_model.transformer.wte.weight.data.to(llama_model.device)

def copy_layer_weights(llama_layer, exaone_layer, device):
    """Copy weights for a single layer from EXAONE to Llama model."""
    # Self-attention
    llama_layer.self_attn.q_proj.weight.data = exaone_layer.attn.attention.q_proj.weight.data.to(device)
    llama_layer.self_attn.k_proj.weight.data = exaone_layer.attn.attention.k_proj.weight.data.to(device)
    llama_layer.self_attn.v_proj.weight.data = exaone_layer.attn.attention.v_proj.weight.data.to(device)
    llama_layer.self_attn.o_proj.weight.data = exaone_layer.attn.attention.out_proj.weight.data.to(device)
    # MLP
    llama_layer.mlp.gate_proj.weight.data = exaone_layer.mlp.c_fc_0.weight.data.to(device)
    llama_layer.mlp.up_proj.weight.data = exaone_layer.mlp.c_fc_1.weight.data.to(device)
    llama_layer.mlp.down_proj.weight.data = exaone_layer.mlp.c_proj.weight.data.to(device)
    # Layer Norms
    llama_layer.input_layernorm.weight.data = exaone_layer.ln_1.weight.data.to(device)
    llama_layer.post_attention_layernorm.weight.data = exaone_layer.ln_2.weight.data.to(device)

def copy_final_weights(llama_model, exaone_model):
    """Copy final layer norm and LM head weights from EXAONE to Llama model."""
    llama_model.model.norm.weight.data = exaone_model.transformer.ln_f.weight.data.to(llama_model.device)
    llama_model.lm_head.weight.data = exaone_model.lm_head.weight.data.to(llama_model.device)

def port_exaone_to_llama(exaone_model_path, llama_model_path):
    print("Loading EXAONE model and tokenizer...")
    exaone_model = AutoModelForCausalLM.from_pretrained(exaone_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    exaone_tokenizer = AutoTokenizer.from_pretrained(exaone_model_path, trust_remote_code=True)
    exaone_config = exaone_model.config

    print("Creating Llama configuration...")
    llama_config = create_llama_config(exaone_config)

    print("Initializing Llama model...")
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Copying weights...")
    copy_embedding_weights(llama_model, exaone_model)

    for i in tqdm(range(exaone_config.num_layers), desc="Copying layers"):
        copy_layer_weights(llama_model.model.layers[i], exaone_model.transformer.h[i], llama_model.device)

    copy_final_weights(llama_model, exaone_model)

    print("Unloading EXAONE model to free memory...")
    unload_model(exaone_model)

    print(f"Saving ported Llama model and tokenizer to {llama_model_path}")
    llama_model.save_pretrained(llama_model_path, safe_serialization=True, max_shard_size="5GB")
    exaone_tokenizer.save_pretrained(llama_model_path)

    print("Unloading Llama model...")
    unload_model(llama_model)

    print(f"EXAONE model successfully ported to Llama format and saved at {llama_model_path}")


if __name__ == "__main__":
    exaone_model_path = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
    llama_model_path = "./exa_llamafied"
    port_exaone_to_llama(exaone_model_path, llama_model_path)
