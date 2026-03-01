FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

RUN pip install --no-cache-dir sam3@git+https://github.com/facebookresearch/sam3.git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
