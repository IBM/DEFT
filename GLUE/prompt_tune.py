import argparse
import os
import random
import sys

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoConfig,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          EvalPrediction, PretrainedConfig,
                          default_data_collator,
                          get_linear_schedule_with_warmup)
from peft import (
    get_peft_model,
    PromptTuningConfig,

)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def setup_seed(seed):
    # seed = cfg.SEED + utils.get_rank() + 10
    print("Setting the Seed to ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Define argparse here
parser = argparse.ArgumentParser()

parser.add_argument("--num_virtual_tokens", type=int, default=60)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_l", type=float, default=1.0)
parser.add_argument("--model_name_or_path", type=str, default="")
parser.add_argument("--task", type=str, default="sst2")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--sparse_obj", action='store_true')
parser.add_argument("--cache_dir", type=str, default="/hf")
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--pad_to_max_length", action="store_true")
parser.add_argument("--eps", type=float, default=20.0)
parser.add_argument("--act_layers_to_save", type=str, default="intermediate_act_fn")
args = parser.parse_args()
print(args)

batch_size = args.batch_size
model_name_or_path =args.model_name_or_path 
task = args.task

device = "cuda"
num_epochs = args.epochs
SEED = args.seed
weight_l = args.weight_l
lr = args.lr

sparse_obj = args.sparse_obj

print("*"*30)
setup_seed(SEED)


if args.task is not None:
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        "glue",
        args.task,
        cache_dir=args.cache_dir,
        use_auth_token= None,
    )

    # Labels
if args.task is not None:
    is_regression = args.task == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    use_auth_token= None,
    padding_side=padding_side
)

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

# Preprocessing the raw_datasets
if args.task is not None:
    sentence1_key, sentence2_key = task_to_keys[args.task]

# Padding strategy
if args.pad_to_max_length:
    padding =  'max_length'#"max_length"
else:
    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    padding = False



peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=args.num_virtual_tokens)



config = AutoConfig.from_pretrained(args.model_name_or_path,num_labels=num_labels, finetuning_task=args.task)
print(config)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,config=config,cache_dir =args.cache_dir)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("PROMPT ARGS: ",peft_config)

print(model)


# Some models have set the order of the labels to use, so let's make sure we do use it.
label_to_id = None
if (
    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    and args.task is not None
    and not is_regression
):
    # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if sorted(label_name_to_id.keys()) == sorted(label_list):
        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
        print(
            "Your model seems to have been trained with labels, but they don't match the dataset: ",
            f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
            "\nIgnoring the model labels as a result.",
        )
elif args.task is None and not is_regression:
    label_to_id = {v: i for i, v in enumerate(label_list)}

if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
elif args.task is not None and not is_regression:
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {id: label for label, id in config.label2id.items()}

if args.max_seq_length > tokenizer.model_max_length:
    print(
        f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # result = tokenizer(*args, max_length=None, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result


raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on dataset",
    remove_columns=["idx"],

)

if "train" not in raw_datasets:
    raise ValueError("--do_train requires a train dataset")
train_dataset = raw_datasets["train"]

if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
    raise ValueError("--do_eval requires a validation dataset")
eval_dataset = raw_datasets["validation_matched" if args.task == "mnli" else "validation"]


class OutputHook(list):

    """ Hook to capture module outputs.

    """

    def __call__(self, module, input, output):

        self.append(output)

output_hook = OutputHook()


for name, module in model.named_modules():
    print(name)
    
act_layers_to_save = args.act_layers_to_save

for name, module in model.named_modules():
    if act_layers_to_save in name:
        hook = module.register_forward_hook(
            output_hook
        )

def compute_l0_penalty(output_hook,eps=1e-7):
    not_zeros = []
    for output in output_hook:
        n_zeros = output**2/(output**2 + eps)
        n_zeros = n_zeros.mean(dim=[1,2])
        n_zeros = n_zeros.reshape(output.shape[0],1)
        not_zeros.append(n_zeros)

    non_zeros = torch.cat(not_zeros,dim=1)

    return non_zeros
    
def compute_eval_sparsity(output_hook):
    not_zeros = []
    for output in output_hook:
        n_zeros = torch.count_nonzero(output,dim=[1,2])/(output.shape[1]*output.shape[2])
        n_zeros = n_zeros.reshape(output.shape[0],1)
        not_zeros.append(n_zeros)

    non_zeros = torch.cat(not_zeros,dim=1)

    return non_zeros


# Get the metric function
if args.task is not None:
    metric = evaluate.load("glue", args.task,experiment_id= f"{args.task}_{args.seed}__{args.sparse_obj}_l0_adapter")
elif is_regression:
    metric = evaluate.load("mse",experiment_id= f"{args.task}_{args.seed}__{args.sparse_obj}_l0_adapter")
else:
    metric = evaluate.load("accuracy",experiment_id= f"{args.task}_{args.seed}__{args.sparse_obj}_l0_adapter")

# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result

# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
# we already did the padding.
if args.pad_to_max_length:
    data_collator = default_data_collator
else:
    data_collator = None



# Instantiate dataloaders.
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size)
eval_dataloader = DataLoader(
    eval_dataset,shuffle=False, collate_fn=data_collator, batch_size=batch_size
)


optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable Parameters: ",trainable_params)
# also print % 
total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters: ",total_params)
print("Trainable %: ",trainable_params/total_params * 100) 

acc_list = []
best_acc = 0.0
model.to(device)
all_dense = []
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        # put every item in batch dictionary to device
     
        # batch = {k: v.to(device)  for k, v in batch.items() if k!='idx'}
        
        batch = {k: v.to(device)  for k, v in batch.items()}
        # print(batch)

        outputs = model(**batch)
        loss = outputs.loss

        if sparse_obj:
            not_zeros = compute_l0_penalty(output_hook,eps=args.eps)
            sparsity_loss = not_zeros.mean()#.sum()
            # print(sparsity_loss)
            total_loss = weight_l*sparsity_loss + loss
        else:
            total_loss = loss
        output_hook.clear()

        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    all_batches_zero = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        # batch = {k: v.to(device)  for k, v in batch.items() if k!='idx'}
        with torch.no_grad():
            outputs = model(**batch)
            not_zeros_Eval = compute_eval_sparsity(output_hook)
       
        all_batches_zero.append(not_zeros_Eval)
        output_hook.clear()

        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    all_batches_zero = torch.cat(all_batches_zero,dim=0)
    print("all_batch_zero SHAPE: ",all_batches_zero.shape)
    mean_outs = all_batches_zero.mean(dim=0)
    print("Approximate Non Zeros: ",mean_outs)
    print("Approximate DENSITYx100 ",mean_outs.mean()*100)
    all_dense.append(mean_outs.mean()*100)

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)
    if sparse_obj:
        model.base_model.save_pretrained(f"/{args.model_name_or_path}/GELU__SPARSE_{weight_l}_{SEED}_{task}_{args.eps}/{task}_sparse_prompt_t_basemodel_{epoch}", from_pt=True)
        model.save_pretrained(f"/{args.model_name_or_path}/GELU__SPARSE_{weight_l}_{SEED}_{task}_{args.eps}/{task}_sparse_prompt_t_{epoch}", from_pt=True)
    else:
        model.base_model.save_pretrained(f"/{args.model_name_or_path}/GELU__{weight_l}_{SEED}_{task}_{args.eps}/{task}_normal_prompt_t_basemodel_{epoch}", from_pt=True)
        model.save_pretrained(f"/{args.model_name_or_path}/GELU__{weight_l}_{SEED}_{task}_{args.eps}/{task}_normal_prompt_t_{epoch}", from_pt=True)
    
        
print("all Density: ",all_dense)
print(acc_list)

print("*"*30)