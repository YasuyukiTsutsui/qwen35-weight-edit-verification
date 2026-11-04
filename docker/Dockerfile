FROM vllm/vllm-openai:v0.32.0
RUN pip install --no-cache-dir "math-verify[antlr4_13_2]==0.9.0"
ENTRYPOINT []
WORKDIR /work
RUN pip install --no-cache-dir "bitsandbytes>=0.48" opencv-python-headless soundfile librosa
