FROM python:3.9-bullseye

COPY . . 

RUN pip install -r requirements.txt

ENV TYPE='system'
ENV SYSTEM='rrf'
ENV METHOD='ranx'


ENTRYPOINT ["python","./run.py"]
