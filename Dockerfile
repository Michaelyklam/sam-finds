FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir sam3@git+https://github.com/facebookresearch/sam3.git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ARG PADDLE_GPU_WHL_URL=https://paddle-whl.bj.bcebos.com/stable/cu129/paddlepaddle-gpu/paddlepaddle_gpu-3.2.2-cp311-cp311-linux_x86_64.whl
RUN pip uninstall -y paddlepaddle && \
    pip install --no-cache-dir "${PADDLE_GPU_WHL_URL}"

# Pre-download model weights from public mirror (avoids gated repo auth at runtime)
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('1038lab/sam3', 'sam3.pt', local_dir='/models')"

COPY app/ app/
COPY test.html test.html

ENV SAM3_CHECKPOINT=/models/sam3.pt
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
