import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# from utils import EarlyStopping, prepare_training_data, target_id_map
from joblib import Parallel, delayed
from sklearn import metrics
from torch.nn import functional as F
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=False, default=0)
    parser.add_argument("--model", type=str, required=False, default="microsoft/deberta-base")
    parser.add_argument("--lr", type=float, required=False, default=3e-5)
    parser.add_argument("--output", type=str, default=".", required=False)
    parser.add_argument("--input", type=str, default="../input", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=2, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=16, required=False)
    parser.add_argument("--epochs", type=int, default=5, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--predict", action="store_true", required=False)
    return parser.parse_args()


def _prepare_training_data_helper(args, tokenizer, df, is_train):
    training_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        idx = row["essay_id"]
        discourse_text = row["discourse_text"]
        discourse_type = row["discourse_type"]

        if is_train:
            filename = os.path.join(args.input, "train", idx + ".txt")
        else:
            filename = os.path.join(args.input, "test", idx + ".txt")

        with open(filename, "r") as f:
            text = f.read()

        encoded_text = tokenizer.encode_plus(
            discourse_type + " " + discourse_text,
            text,
            add_special_tokens=False,
        )
        input_ids = encoded_text["input_ids"]

        sample = {
            "discourse_id": row["discourse_id"],
            "input_ids": input_ids,
            # "discourse_text": discourse_text,
            # "essay_text": text,
            # "mask": encoded_text["attention_mask"],
        }

        if "token_type_ids" in encoded_text:
            sample["token_type_ids"] = encoded_text["token_type_ids"]

        label = row["discourse_effectiveness"]

        sample["label"] = LABEL_MAPPING[label]

        training_samples.append(sample)
    return training_samples


def prepare_training_data(df, tokenizer, args, num_jobs, is_train):
    training_samples = []

    df_splits = np.array_split(df, num_jobs)

    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_training_data_helper)(args, tokenizer, df, is_train) for df in df_splits
    )
    for result in results:
        training_samples.extend(result)

    return training_samples


class FeedbackDataset:
    def __init__(self, samples, args, tokenizer):
        self.samples = samples
        self.args = args
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]["input_ids"]
        label = self.samples[idx]["label"]

        input_ids = [self.tokenizer.cls_token_id] + ids

        if len(input_ids) > self.args.max_len - 1:
            input_ids = input_ids[: self.args.max_len - 1]

        input_ids = input_ids + [self.tokenizer.sep_token_id]
        mask = [1] * len(input_ids)

        return {
            "ids": input_ids,
            "mask": mask,
            # "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": label,
        }


class Collate:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        output = dict()
        output["ids"] = [sample["ids"] for sample in batch]
        output["mask"] = [sample["mask"] for sample in batch]
        output["targets"] = [sample["targets"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
            output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
        else:
            output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
            output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]

        # convert to tensors
        output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        output["mask"] = torch.tensor(output["mask"], dtype=torch.long)
        output["targets"] = torch.tensor(output["targets"], dtype=torch.long)

        return output


class AttentionHead(nn.Module):
    def __init__(self, in_size: int = 768, hidden_size: int = 512) -> None:
        super().__init__()
        self.W = nn.Linear(in_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        output = self.dropout(context_vector)
        return output


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class FeedbackModel(nn.Module):
    def __init__(self, model_name, num_train_steps, learning_rate, num_labels, steps_per_epoch):
        super().__init__()
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.steps_per_epoch = steps_per_epoch

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        # self.attention = AttentionHead(in_size=config.hidden_size, hidden_size=config.hidden_size)
        self.pooler = MeanPooling()
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def optimizer_scheduler(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=self.learning_rate)
        sch = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=self.num_train_steps,
            last_epoch=-1,
        )
        # sch = get_cosine_schedule_with_warmup(
        #     opt,
        #     num_warmup_steps=0,  # int(0.1 * self.num_train_steps),
        #     num_training_steps=self.num_train_steps,
        #     num_cycles=1,
        #     last_epoch=-1,
        # )
        return opt, sch

    def loss(self, outputs, targets):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, targets)
        return loss

    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        # print(outputs)
        # print(targets)
        mll = metrics.log_loss(
            targets.cpu().detach().numpy(),
            outputs.cpu().detach().numpy(),
            labels=[0, 1, 2],
        )
        return {"mll": torch.tensor(mll, device=device)}

    def forward(self, ids, mask, token_type_ids=None, targets=None):

        if token_type_ids:
            transformer_out = self.transformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.transformer(ids, mask)
        # sequence_output = transformer_out.pooler_output
        sequence_output = transformer_out.last_hidden_state
        # sequence_output = self.attention(sequence_output)
        sequence_output = self.pooler(sequence_output, mask)
        sequence_output = self.dropout(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        logits = torch.softmax(logits, dim=-1)
        loss = 0

        if targets is not None:
            loss1 = self.loss(logits1, targets)
            loss2 = self.loss(logits2, targets)
            loss3 = self.loss(logits3, targets)
            loss4 = self.loss(logits4, targets)
            loss5 = self.loss(logits5, targets)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            metric = self.monitor_metrics(logits, targets)
            return logits, loss, metric

        return logits, loss, {}


def main(args):
    NUM_JOBS = 12
    seed_everything(42)
    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(os.path.join(args.input, "train_folds.csv"))

    train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    training_samples = prepare_training_data(train_df, tokenizer, args, num_jobs=NUM_JOBS, is_train=True)
    valid_samples = prepare_training_data(valid_df, tokenizer, args, num_jobs=NUM_JOBS, is_train=True)

    training_samples = list(sorted(training_samples, key=lambda d: len(d["input_ids"])))
    valid_samples = list(sorted(valid_samples, key=lambda d: len(d["input_ids"])))

    train_dataset = FeedbackDataset(training_samples, args, tokenizer)
    valid_dataset = FeedbackDataset(valid_samples, args, tokenizer)

    num_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)

    collate_fn = Collate(tokenizer, args)

    model = FeedbackModel(
        model_name=args.model,
        num_train_steps=num_train_steps,
        learning_rate=args.lr,
        num_labels=3,
        steps_per_epoch=len(train_dataset) / args.batch_size,
    )

    model = Tez(model)
    es = EarlyStopping(
        monitor="valid_loss",
        model_path=os.path.join(args.output, f"model_f{args.fold}.bin"),
        patience=5,
        mode="min",
        delta=0.001,
        save_weights_only=True,
    )
    config = TezConfig(
        training_batch_size=args.batch_size,
        validation_batch_size=args.valid_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        fp16=True,
        step_scheduler_after="batch",
        val_strategy="batch",
        val_steps=1500,
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_collate_fn=collate_fn,
        valid_collate_fn=collate_fn,
        callbacks=[es],
        config=config,
    )


def predict(args):
    NUM_JOBS = 2
    seed_everything(42)
    df = pd.read_csv(os.path.join(args.input, "test.csv"))
    df.loc[:, "discourse_effectiveness"] = "Adequate"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = prepare_training_data(df, tokenizer, args, num_jobs=NUM_JOBS, is_train=False)
    samples = list(sorted(samples, key=lambda d: len(d["input_ids"])))

    dataset = FeedbackDataset(samples, args, tokenizer)
    num_train_steps = int(len(dataset) / args.batch_size / args.accumulation_steps * args.epochs)

    model = FeedbackModel(
        model_name=args.model,
        num_train_steps=num_train_steps,
        learning_rate=args.lr,
        num_labels=3,
        steps_per_epoch=len(dataset) / args.batch_size,
    )

    model = Tez(model)
    config = TezConfig(
        test_batch_size=args.batch_size,
        fp16=True,
    )
    model.load(os.path.join(args.output, f"model_f{args.fold}.bin"), weights_only=True, config=config)

    collate_fn = Collate(tokenizer, args)
    iter_preds = model.predict(dataset, collate_fn=collate_fn)
    preds = []
    for temp_preds in iter_preds:
        preds.append(temp_preds)

    preds = np.vstack(preds)

    sample_submission = pd.read_csv(os.path.join(args.input, "sample_submission.csv"))
    sample_submission.loc[:, "discourse_id"] = [x["discourse_id"] for x in samples]
    sample_submission.loc[:, "Ineffective"] = preds[:, 0]
    sample_submission.loc[:, "Adequate"] = preds[:, 1]
    sample_submission.loc[:, "Effective"] = preds[:, 2]
    sample_submission.to_csv(f"preds_{args.fold}.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.predict:
        predict(args)
    else:
        main(args)

