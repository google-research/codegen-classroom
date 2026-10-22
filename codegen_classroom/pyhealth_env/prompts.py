"""Student prompts for PyHealth environments."""

from __future__ import annotations

import textwrap

from codegen_classroom import scoring
from codegen_classroom.pyhealth_env import prepare as prepare_lib
from codegen_classroom.pyhealth_env import spec as spec_lib

SYSTEM_INSTRUCTION = """\
You are an expert clinical machine learning engineer who writes correct,
efficient PyTorch / PyHealth 2.0 code. You iteratively improve a model for a
clinical prediction task based on validation scores and execution feedback.

Always answer with a short plan followed by ONE complete ```python code block
that defines the requested function. The code is run as a standalone module, so
include every import it needs. Do not read files outside the provided datasets,
do not download data, and do not use the network.
"""

PYHEALTH_CHEATSHEET = """\
PyHealth 2.0 quick reference:
- Datasets: `train_dataset` / `val_dataset` are `pyhealth.datasets.SampleDataset`
  subsets. Each sample is a dict of processed tensors keyed by the task's input
  and output schema (see below), plus `patient_id`.
- Loaders: `from pyhealth.datasets import get_dataloader`;
  `get_dataloader(ds, batch_size=64, shuffle=True)` yields dict batches.
- Models (`from pyhealth.models import ...`): RNN, Transformer, RETAIN, AdaCare,
  ConCare, StageNet, TCN, MLP, CNN, LogisticRegression, ... All take
  `dataset=train_dataset` as the first argument (it carries the fitted feature
  processors and label schema) plus model hyperparameters, e.g.
  `RNN(dataset=train_dataset, embedding_dim=128, hidden_dim=128)`.
  `model(**batch)` returns a dict with "loss", "y_prob", "y_true", "logit".
- Training: `from pyhealth.trainer import Trainer`;
  `trainer = Trainer(model=model, device=device, metrics=[...],
  output_path="trainer_output")`; `trainer.train(train_dataloader=...,
  val_dataloader=..., epochs=..., monitor=..., optimizer_params={"lr": 1e-3},
  weight_decay=..., patience=...)`. With a val loader and `monitor`, the best
  checkpoint is reloaded at the end, but ONLY if logging is enabled (the
  default; checkpoints go to `output_path`, inside your working directory).
  `enable_logging=False` silently keeps the last epoch instead. Binary metrics include "roc_auc", "pr_auc", "f1";
  multiclass metrics include "accuracy", "f1_macro", "cohen_kappa".
- You may also write a custom `torch.nn.Module` (e.g. subclass
  `pyhealth.models.BaseModel`) as long as `model(**batch)` returns a dict with
  "y_prob" (and ideally "y_true") for a batch from `get_dataloader`.
"""

REFINE_INSTRUCTIONS = (
    "Improve your solution. Consider a stronger or different architecture"
    " (e.g. Transformer, RETAIN, StageNet, AdaCare) or tuning its capacity.",
    "Improve generalization: tune dropout, weight decay, learning rate and the"
    " number of epochs, and use early stopping on the validation set.",
    "Address class imbalance (e.g. a weighted loss or sampling) if the label"
    " distribution is skewed, and make sure the monitored metric matches the"
    " scoring metrics.",
    "Use the execution output to find inefficiencies or bugs. Make sure"
    " training fits comfortably within the time budget.",
    "Try combining features differently or ensembling a few small models,"
    " keeping the total runtime within the time budget.",
)


# A working custom-model solution: shows how to wrap a model in
# pyhealth.models.BaseModel and train it with a plain PyTorch loop. Students
# are shown it as a template; tests run it end to end.
BASEMODEL_TEMPLATE = """\
import copy

import numpy as np
import torch
import torch.nn as nn
from pyhealth.datasets import get_dataloader
from pyhealth.models import BaseModel
from sklearn.metrics import average_precision_score, roc_auc_score


class BagOfCodesModel(BaseModel):
    \"\"\"Embeds each token-sequence feature, mean-pools it and classifies.\"\"\"

    def __init__(self, dataset, embedding_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__(dataset)  # Sets feature_keys, label_keys, mode.
        self.label_key = self.label_keys[0]
        self.embeddings = nn.ModuleDict()
        for key in self.feature_keys:
            processor = dataset.input_processors[key]
            # Token features: integer ids in [0, processor.size()), 0 = padding.
            self.embeddings[key] = nn.Embedding(
                processor.size(), embedding_dim, padding_idx=0
            )
        self.head = nn.Sequential(
            nn.Linear(len(self.feature_keys) * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.get_output_size()),
        )

    def forward(self, **batch):
        pooled = []
        for key in self.feature_keys:
            x = batch[key].to(self.device).long()
            x = x.reshape(x.shape[0], -1)  # Flatten nested sequences: (B, L).
            mask = (x != 0).unsqueeze(-1).float()
            emb = self.embeddings[key](x) * mask
            pooled.append(emb.sum(1) / mask.sum(1).clamp(min=1.0))
        logits = self.head(torch.cat(pooled, dim=-1))
        y_true = batch[self.label_key].to(self.device)
        return {
            "loss": self.get_loss_function()(logits, y_true),
            "y_prob": self.prepare_y_prob(logits),
            "y_true": y_true,
            "logit": logits,
        }


def _evaluate(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            probs.append(out["y_prob"].cpu().numpy())
            labels.append(out["y_true"].cpu().numpy())
    y_prob, y_true = np.concatenate(probs), np.concatenate(labels)
    if model.mode == "binary":
        y_prob, y_true = y_prob.ravel(), y_true.ravel()
        return {"auroc": roc_auc_score(y_true, y_prob),
                "auprc": average_precision_score(y_true, y_prob)}
    return {"accuracy": float((y_prob.argmax(1) == y_true.ravel()).mean())}


def solution(train_dataset, val_dataset, device, epochs=5):
    train_loader = get_dataloader(train_dataset, batch_size=128, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=512, shuffle=False)
    model = BagOfCodesModel(train_dataset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score, best_state = -np.inf, None
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            model(**batch)["loss"].backward()
            optimizer.step()
        metrics = _evaluate(model, val_loader)
        score = float(np.mean(list(metrics.values())))
        print(f"epoch {epoch}: {metrics}")
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model
"""


def _monitor_metric(spec: spec_lib.EnvSpec) -> str:
  return "roc_auc" if spec.task_type == spec_lib.BINARY else "accuracy"


def _feature_table(prepared: prepare_lib.PreparedEnv) -> str:
  """Markdown table of each batch field: processor, vocab and tensor layout."""
  inputs = prepared.schema_summary.get("inputs", {})
  outputs = prepared.schema_summary.get("outputs", {})
  rows = [
      "| batch key | role | processor | vocab/classes | batch value (B=4) |",
      "|---|---|---|---|---|",
  ]
  for key, info in prepared.batch_summary.items():
    meta = inputs.get(key) or outputs.get(key) or {}
    role = (
        "input feature"
        if key in inputs
        else "label" if key in outputs else "metadata"
    )
    if "shape" in info:
      layout = f"{info['type']} {info['dtype']} {info['shape']}"
    elif "shapes" in info:
      layout = f"{info['type']} {info['shapes']} {info['dtypes']}"
    else:
      layout = info["type"]
    token = " (token ids, 0 = padding)" if meta.get("is_token") else ""
    rows.append(
        f"| `{key}` | {role} | {meta.get('processor', '-')}{token} |"
        f" {meta.get('size', '-')} | {layout} |"
    )
  return "\n".join(rows)


def build_task_prompt(
    spec: spec_lib.EnvSpec, prepared: prepare_lib.PreparedEnv
) -> str:
  """The initial student prompt describing the environment and contract."""
  sizes = prepared.split_sizes
  labels = prepared.label_distribution
  total = sum(labels.values()) or 1
  label_lines = ", ".join(
      f"{k}: {v / total:.1%}" for k, v in sorted(labels.items())
  )
  y_prob_shape = (
      "shape (B, 1) or (B,): probability of the positive class"
      if spec.task_type == spec_lib.BINARY
      else "shape (B, C): class probabilities"
  )
  return f"""\
# Task: {spec.name}
{spec.description}

Task type: {spec.task_type}. Label field: "{spec.label_key}".
Label distribution (training split sample): {label_lines}.

## What to write
```
def solution(train_dataset, val_dataset, device):
    ...
    return model
```
- `train_dataset`, `val_dataset`: PyHealth `SampleDataset` subsets split by
  patient ({sizes[prepare_lib.LEARNING_TRAIN]} / {sizes[prepare_lib.LEARNING_VAL]} samples during learning cycles; the final
  evaluation reruns your code on the full splits, {sizes[prepare_lib.TRAIN]} / {sizes[prepare_lib.VAL]} samples).
- `device`: "cuda" (one GPU, shared, keep memory use modest) or "cpu".
- Return the trained model. It is evaluated with `model.eval()` and
  `torch.no_grad()` on batches from
  `get_dataloader(eval_dataset, batch_size={spec.inference_batch_size}, shuffle=False)`:
  `model(**batch)` must return a dict with "y_prob" ({y_prob_shape}) and
  ideally "y_true". Learning cycles are scored on the validation split; the
  final score uses a held-out test split you never see.
- Budget: {int(spec.learning_timeout_seconds)}s per learning-cycle run, {int(spec.final_timeout_seconds)}s for the final run
  (training + evaluation), about {spec.epochs_hint} epochs. Runs over budget are killed.
- Print short progress lines (e.g. validation metrics per epoch); they are
  shown back to you.

## Data
Datasets yield dict samples; `get_dataloader(dataset, batch_size, shuffle)`
collates them into dict batches with these fields:

{_feature_table(prepared)}

Token sequences are padded with 0 within a batch; vocab sizes include the
padding and unknown tokens, so `nn.Embedding(processor.size(), dim,
padding_idx=0)` is valid. Fitted processors are available as
`train_dataset.input_processors[key]` / `train_dataset.output_processors[key]`.

## Scoring
{scoring.get_metrics_description(spec.metric_weights)}

{PYHEALTH_CHEATSHEET}
## Template A: built-in PyHealth model + Trainer
{_strip_plan(build_seed_solution(spec))}
## Template B: custom model wrapped in `pyhealth.models.BaseModel`
Subclass `BaseModel` (call `super().__init__(dataset)`; it provides
`feature_keys`, `label_keys`, `mode`, `device`, `get_output_size()`,
`get_loss_function()` and `prepare_y_prob(logits)`) and return the standard
output dict, so the model works with both a custom loop and `Trainer`:

```python
{BASEMODEL_TEMPLATE}```

These templates are starting points: change the architecture, features,
loss, optimization and training schedule to maximize the score.
"""


def _strip_plan(solution_text: str) -> str:
  """Returns only the fenced code block of a seed solution."""
  return solution_text[solution_text.index("```python"):]


def build_seed_solution(spec: spec_lib.EnvSpec) -> str:
  """A working RNN baseline in the solution contract, used as a seed."""
  metrics = (
      '["roc_auc", "pr_auc", "f1"]'
      if spec.task_type == spec_lib.BINARY
      else '["accuracy", "f1_macro"]'
  )
  code = textwrap.dedent(f"""\
      import torch
      from pyhealth.datasets import get_dataloader
      from pyhealth.models import RNN
      from pyhealth.trainer import Trainer


      def solution(train_dataset, val_dataset, device):
          train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
          val_loader = get_dataloader(val_dataset, batch_size=256, shuffle=False)
          model = RNN(
              dataset=train_dataset,
              embedding_dim=128,
              hidden_dim=128,
              num_layers=1,
              dropout=0.3,
          )
          # Logging must stay enabled for the best checkpoint to be reloaded.
          trainer = Trainer(
              model=model,
              device=device,
              metrics={metrics},
              output_path="trainer_output",
          )
          trainer.train(
              train_dataloader=train_loader,
              val_dataloader=val_loader,
              epochs={spec.epochs_hint},
              monitor="{_monitor_metric(spec)}",
              optimizer_params={{"lr": 1e-3}},
          )
          print("val:", trainer.evaluate(val_loader))
          return model
      """)
  return (
      "Plan: start from a GRU baseline (PyHealth RNN) trained with the"
      " PyHealth Trainer, monitoring the validation metric.\n\n"
      f"```python\n{code}```\n"
  )
