import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#####################################
# 学習パラメーター
#####################################
EPOCHS = 1
MAX_STEPS = 2000
LEARNING_RATE = 4e-4
VAL_SET_SIZE = 0.2  # 検証分割比率
CUTOFF_LEN = 512  # コンテキスト長の上限
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
#####################################
# 設定
#####################################
MODEL_NAME = "cyberagent/calm2-7b-chat"
DATASET_NAME = "takaaki-inada/databricks-dolly-15k-ja-zundamon"
SAVE_ID = "zunda01"
TARGET_MODULES = ["q_proj", "v_proj"]  # どれが必要かはprint.py参照
SAVE_STEPS = 100

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
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


def generate_prompt(data_point):
    if data_point["input"]:
        result = f"""USER: {data_point["instruction"]}
{data_point["input"]}

ASSISTANT: {data_point["output"]}<|endoftext|>"""
    else:
        result = f"""USER: {data_point["instruction"]}
ASSISTANT: {data_point["output"]}<|endoftext|>"""
    # result = result.replace("\n", "<NL>")  # 改行→<NL> rinnaの場合必要
    return result


def tokenize(prompt, tokenizer):
    result = tokenizer(prompt, truncation=True, padding=False, max_length=CUTOFF_LEN + 1)
    return {"input_ids": result["input_ids"], "attention_mask": result["attention_mask"]}


data = load_dataset(DATASET_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# 学習データと検証データの準備
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
    val_data = train_val["test"].shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
else:
    train_data = data["train"].shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
    val_data = None

args = transformers.TrainingArguments(
    # per_device_train_batch_size=2,
    # gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    max_steps=MAX_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
    eval_steps=SAVE_STEPS if VAL_SET_SIZE > 0 else None,
    save_steps=SAVE_STEPS,
    # warmup_ratio=0.03,
    warmup_steps=100,
    logging_steps=10,
    output_dir=f"output/{SAVE_ID}",
    lr_scheduler_type="constant",
    report_to="none",
    save_total_limit=10,
    load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
    auto_find_batch_size=True,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

model = torch.compile(model)  # PyTorch 2.0以降 高速化なるらしい
trainer.train()
