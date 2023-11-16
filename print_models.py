from transformers import AutoModelForCausalLM

model_name = "cyberagent/calm2-7b-chat"

model = AutoModelForCausalLM.from_pretrained(model_name)
print(model)
# (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
# (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
# (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
# (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
# (rotary_emb): LlamaRotaryEmbedding()
