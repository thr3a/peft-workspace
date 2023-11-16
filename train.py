import os

import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

#####################################
# 学習パラメーター
#####################################
EPOCHS = 1
MAX_STEPS = 2000
LEARNING_RATE = 4e-4
VAL_SET_SIZE = 0.2 # 検証分割比率
CUTOFF_LEN = 1000  # コンテキスト長の上限
#####################################
# 設定
#####################################
MODEL_NAME = "cyberagent/calm2-7b-chat"
DATASET_NAME = "takaaki-inada/databricks-dolly-15k-ja-zundamon"
SAVE_ID = "zunda01"
TARGET_MODULES = ["q_proj", "v_proj"] # どれが必要かはprint.py参照

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=TARGET_MODULES,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

def generate_prompt(data_point):
    if data_point["input"]:
        result =  f"""USER: {data_point["instruction"]}
{data_point["input"]}

ASSISTANT: {data_point["output"]}<|endoftext|>"""
    else:
        result = f"""USER: {data_point["instruction"]}
ASSISTANT: {data_point["output"]}<|endoftext|>"""
    # result = result.replace('\n', '<NL>') # 改行→<NL>
    return result

def tokenize(prompt, tokenizer):
    # max_length
    result = tokenizer(prompt, truncation=True, padding=False)
    return {"input_ids": result["input_ids"], "attention_mask": result["attention_mask"]}

data = load_dataset(DATASET_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# 学習データと検証データの準備
# train_val = data["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
# train_data = train_val["train"]
# val_data = train_val["test"]
# train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
# val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
data = data['train'].map(lambda x: tokenize(generate_prompt(x), tokenizer))

# old
# data = data.map(lambda samples: tokenizer(samples["output"]), batched=True)

args = transformers.TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    # warmup_ratio=0.03,
    warmup_steps=100,
    max_steps=MAX_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    # evaluation_strategy="steps",
    # eval_steps=100,
    save_steps=100,
    logging_steps=10,
    output_dir=f"output/{SAVE_ID}",
    lr_scheduler_type="constant",
    report_to="none",
    save_total_limit=10,
    # auto_find_batch_size=True,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    # eval_dataset = val_data,
    args=args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

model = torch.compile(model) # PyTorch 2.0以降 高速化
trainer.train()
