FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

COPY . .
RUN pip install -r requirements.txt

ENV METHOD='e5'
ENV TYPE='index'

ENV PYTHONUNBUFFERED=1

ENV TRANSFORMERS_CACHE=/data/model/transformer-cache/
ENV IR_DATASETS_HOME=/data/dataset/ir_datasets-cache/

ENTRYPOINT ["python","./run.py"]
