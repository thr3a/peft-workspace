from transformers import AutoModelForCausalLM

model_name = "cyberagent/calm2-7b-chat"

model = AutoModelForCausalLM.from_pretrained(model_name)
print(model)
# cyberagent/calm2-7b-chat
# LlamaForCausalLM(
#   (model): LlamaModel(
#     (embed_tokens): Embedding(65024, 4096, padding_idx=1)
#     (layers): ModuleList(
#       (0-31): 32 x LlamaDecoderLayer(
#         (self_attn): LlamaAttention(
#           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (rotary_emb): LlamaRotaryEmbedding()
#         )
#         (mlp): LlamaMLP(
#           (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
#           (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
#           (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
#           (act_fn): SiLUActivation()
#         )
#         (input_layernorm): LlamaRMSNorm()
#         (post_attention_layernorm): LlamaRMSNorm()
#       )
#     )
#     (norm): LlamaRMSNorm()
#   )
#   (lm_head): Linear(in_features=4096, out_features=65024, bias=False)
# )
