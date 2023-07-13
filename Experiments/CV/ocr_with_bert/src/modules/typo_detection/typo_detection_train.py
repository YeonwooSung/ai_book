import os
from datetime import datetime as dtime

import numpy as np
import wandb
import torch
import torch.utils.data as torchdata
from torch import nn
from sklearn.metrics import accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


def prepare_dataloader(inputs_path: str,
                       labels_path: str,
                       masks_path: str,
                       batch_size: int = 32,
                       mode: str = 'train'
                       ):
    assert mode in ['train', 'val'], 'Wrong dataloader type'
    
    inputs = torch.load(inputs_path)
    labels = torch.load(labels_path)
    masks = torch.load(masks_path)
    
    dataset = torchdata.TensorDataset(inputs, masks, labels)
    
    if mode == 'train':
        sampler = torchdata.RandomSampler(dataset)
    else:
        sampler = torchdata.SequentialSampler(dataset)

    dataloader = torchdata.DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    return dataloader

def prepare_optimizer(model,
                      lr: float = 3e-5,
                      eps: float = 1e-8
                      ):
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if "bias" in n],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if "bias" not in n],
        'weight_decay_rate': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=lr,
                      eps=eps)

    return optimizer


def train_detector(model,
                   tokenizer,
                   dataloaders,
                   output_dir: str, 
                   epochs: int,
                   lr: float,
                   wandb_log: bool = False
                   ):
    time_now_str  = dtime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f'ocr-with-bert-{time_now_str}'
    
    if wandb_log:
        wandb.init(run_name)
        config = wandb.config
        config.learning_rate = lr
        config.epochs = epochs
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_fct = nn.CrossEntropyLoss()
    #loss_fct = nn.BCELoss()
    
    optimizer = prepare_optimizer(model,
                                  lr=lr)
    
    total_steps = len(dataloaders['train']) * dataloaders['train'].batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=2,
                                                num_training_steps=total_steps)
    
    loss_arr = {'train': [], 'val': []}
    acc_arr = {'train': [], 'val': []}

    best_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        
        for mode in ['train', 'val']:
            loss_epoch = 0
            predictions, true_labels = [], []
            
            if mode == 'train':
                model.train()              
            else:
                model.eval()
                
            for step, batch in enumerate(tqdm(dataloaders[mode])):
                inputs, masks, labels = tuple(t.to(device) for t in batch)
                
                if mode == 'val':
                    with torch.no_grad():
                        logits = model(inputs, attention_mask=masks)
                else:
                    logits = model(inputs, attention_mask=masks)
                
                active_loss = masks.view(-1) == 1
                active_logits = logits.view(-1, model.num_labels)
                active_labels = torch.where(active_loss, labels.view(-1),
                                            torch.tensor(loss_fct.ignore_index).type_as(labels))

                                
                loss = loss_fct(active_logits, active_labels)
                loss_epoch += loss.item()
                
                logits = logits.detach().cpu().numpy()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(labels.cpu().tolist())
                
                if mode == 'train':
                    model.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=2.0)
                    optimizer.step()
                    scheduler.step()
                
                if wandb_log and ((step % 1000) == 0):
                    wandb.log({f"loss_{mode}" : loss})
  
            loss_avg = loss_epoch / len(dataloaders[mode])
            print(f"{mode} loss: {loss_avg}")
            loss_arr[mode].append(loss_avg)
            
            pred_tags = [p_i for p, l, m in zip(predictions, true_labels, masks)
                         for p_i, l_i, m_i in zip(p, l, m) if m_i]
            true_tags = [l_i for l, m in zip(true_labels, masks)
                         for l_i, m_i in zip(l, m) if m_i]
            acc = accuracy_score(pred_tags, true_tags)
            print(f"{mode} accuracy: {acc}")
            acc_arr[mode].append(acc)
            
            if wandb_log:
                wandb.log({f"accuracy_{mode}": acc})
            
            if mode == 'val':
                if loss_avg < best_loss:
                    best_loss = loss_avg
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(output_dir, "best_model.pth"))
                    tokenizer.save_pretrained(output_dir)
                    
