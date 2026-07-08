from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, AutoModelForSequenceClassification
import torch.nn as nn
import pandas as pd
import torch
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from datasets import Dataset
import warnings
from transformers.utils import logging
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import csv
import random
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import numpy as np
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="PyTorch is not compiled with NCCL support")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

seed = 42
ds = load_dataset("zjunlp/FactCHD")

def join_query_response(query_list, response_list):
    return [q + " [SEP] " + r for q, r in zip(query_list, response_list)]

# Prepare texts
all_texts = join_query_response(ds["train"]["query"], ds["train"]["response"])
all_labels = ds["train"]["label"]

# Train/val split (85/15)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_texts,
    all_labels,
    test_size=0.15,
    stratify=all_labels,
    random_state=seed
)

# Test split
test_texts  = join_query_response(ds["test"]["query"], ds["test"]["response"])
test_labels = ds["test"]["label"]
# Map labels to ints
label2id = {"FACTUAL": 0, "NON-FACTUAL": 1}
id2label = {0: "FACTUAL", 1: "NON-FACTUAL"}

train_labels = [label2id[l] for l in train_labels]
val_labels   = [label2id[l] for l in val_labels]
test_labels  = [label2id[l] for l in test_labels]

print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

# Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -10.0))

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -10.0))

        self.prior_std = prior_std

    def forward(self, x):
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        prior_var = self.prior_std ** 2

        weight_kl = 0.5 * torch.sum(
            (self.weight_mu ** 2 + torch.exp(self.weight_logvar)) / prior_var
            - 1.0
            - self.weight_logvar
            + math.log(prior_var)
        )

        bias_kl = 0.5 * torch.sum(
            (self.bias_mu ** 2 + torch.exp(self.bias_logvar)) / prior_var
            - 1.0
            - self.bias_logvar
            + math.log(prior_var)
        )

        return weight_kl + bias_kl

class BayesianMLP(nn.Module):
    def __init__(self, in_features, num_labels, expansion=4):
        super().__init__()

        hidden = in_features * expansion

        self.fc1 = BayesianLinear(in_features, hidden)
        self.fc2 = BayesianLinear(hidden, num_labels)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    def kl_divergence(self):
        return self.fc1.kl_divergence() + self.fc2.kl_divergence()

class BayesianRoBERTaClassifier(nn.Module):
    def __init__(self, model_name, num_labels, beta=1e-4):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.03)

        # 4× Bayesian Head
        self.classifier = BayesianMLP(
            self.encoder.config.hidden_size,
            num_labels,
            expansion=4
        )

        self.beta = beta

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)

        logits = self.classifier(cls_embedding)

        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
            kl_loss = self.classifier.kl_divergence() / input_ids.size(0)
            loss = ce_loss + self.beta * kl_loss

        return {"loss": loss, "logits": logits}

# Model & Tokenizer
model_name = "FacebookAI/roberta-base"

model = BayesianRoBERTaClassifier(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Freeze RoBERTa encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Print Trainable Params
def print_trainable_parameters(model):
    total = 0
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,}")
            total += param.numel()
    print(f"\nTotal trainable parameters: {total:,}\n")

print_trainable_parameters(model)

# Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "mcc": matthews_corrcoef(labels, preds)
    }

# Metric Recorder Callback
epoch_metrics = []

class MetricRecorderCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch_metrics.append(metrics)
        return control

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    num_train_epochs=12,
    logging_steps=10,
    logging_dir="./logs",
    load_best_model_at_end=True,
    remove_unused_columns=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        MetricRecorderCallback(),
        EarlyStoppingCallback(early_stopping_patience=2)
    ]
)

trainer.train()

# Standard Evaluation
results = trainer.evaluate(test_dataset)
print("\nTest results:", results)

# Bayesian MC Prediction
def bayesian_predict(trainer, dataset, mc_samples=20):
    model = trainer.model
    model.eval()

    probs_all = []

    for _ in range(mc_samples):
        outputs = trainer.predict(dataset)
        logits = torch.tensor(outputs.predictions)
        probs = torch.softmax(logits, dim=-1)
        probs_all.append(probs.unsqueeze(0))

    probs_all = torch.cat(probs_all, dim=0)

    mean_probs = probs_all.mean(dim=0)
    uncertainty = probs_all.var(dim=0).mean(dim=1)
    preds = mean_probs.argmax(dim=-1)

    return preds.numpy(), mean_probs.numpy(), uncertainty.numpy()

preds, probs, uncertainty = bayesian_predict(trainer, test_dataset)

# Confusion Matrix
cm = confusion_matrix(test_labels, preds)
print("\nConfusion Matrix:")
print(cm)
