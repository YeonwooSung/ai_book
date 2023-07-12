import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def train(
    model_name: str = "Salesforce/xgen-7b-8k-base",
    dataset_name: str = "tatsu-lab/alpaca",
    output_dir: str = "./results",
    load_model_in_4bit: bool = True,
    model_dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    lora_peft_r: int = 16,
    lora_peft_alpha: int = 32,
    lora_peft_dropout: float = 0.05,
    lora_peft_bias: str = "none",
    lora_peft_task_type: str = "CAUSAL_LM",
    per_device_train_batch_size: int = 4,
    optim: str = "adamw_torch",
    logging_steps: int = 100,
    learning_rate: float = 2e-4,
    use_fp16: bool = True,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "linear",
    num_train_epochs: int = 1,
    save_strategy: str = "epoch",
    push_to_hub: bool = False,
    dataset_text_field: str = "text",
    max_seq_length: int = 1024,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert model_dtype in [torch.float16, torch.float32]
    assert device_map in ["auto", "cpu", "cuda"]

    if use_fp16 or load_model_in_4bit:
        if not torch.cuda.is_available():
            print('fp16 is only available on cuda devices, use fp32 instead')
            use_fp16 = False
            load_model_in_4bit = False

    train_dataset = load_dataset(dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.unk_token_id is None:
    #     tokenizer.unk_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=load_model_in_4bit,
        torch_dtype=model_dtype,
        device_map=device_map,
    )
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_int8_training(
        model,
        use_gradient_checkpointing=True,
    )
    peft_config = LoraConfig(
        r=lora_peft_r,
        lora_alpha=lora_peft_alpha,
        lora_dropout=lora_peft_dropout,
        bias=lora_peft_bias,
        task_type=lora_peft_task_type,
    )
    peft_model = get_peft_model(model, peft_config)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        optim=optim,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=use_fp16,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        num_train_epochs=num_train_epochs,
        save_strategy=save_strategy,
        push_to_hub=push_to_hub,
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_dataset,
        dataset_text_field=dataset_text_field,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        peft_config=peft_config,
    )
    trainer.train()
    if push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    train(
        model_name="Salesforce/xgen-7b-8k-base",
        dataset_name="tatsu-lab/alpaca",
        output_dir="./results",
        load_model_in_4bit=True,
        model_dtype=torch.float16,
        device_map="auto",
        lora_peft_r=16,
        lora_peft_alpha=32,
        lora_peft_dropout=0.05,
        lora_peft_bias="none",
        lora_peft_task_type="CAUSAL_LM",
        per_device_train_batch_size=4,
        optim="adamw_torch",
        logging_steps=100,
        learning_rate=2e-4,
        use_fp16=True,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        num_train_epochs=1,
        save_strategy="epoch",
        push_to_hub=False,
        dataset_text_field="text",
        max_seq_length=1024
    )
