FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir sam3@git+https://github.com/facebookresearch/sam3.git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights from public mirror (avoids gated repo auth at runtime)
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('1038lab/sam3', 'sam3.pt', local_dir='/models')"

COPY app/ app/
COPY test.html test.html

ENV SAM3_CHECKPOINT=/models/sam3.pt
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
