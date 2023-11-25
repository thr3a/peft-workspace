FROM --platform=linux/x86_64 ghcr.io/thr3a/peft-workspace:latest

WORKDIR /app
COPY ./requirements.txt ./
RUN pip install packaging
RUN pip install -r requirements.txt
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('rinna/japanese-gpt-neox-3.6b-instruction-ppo')"
# ENV HF_HUB_OFFLINE=1

# CMD ["python", "app.py", "--input_audio_max_duration", "-1", "--server_name", "0.0.0.0", "--auto_parallel", "True", "--default_model_name", "large-v2"]
