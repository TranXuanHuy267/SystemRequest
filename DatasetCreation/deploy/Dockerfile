
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# RUN apt update
# RUN apt install -y git

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./model_matching /code/model_matching

COPY ./model /code/model

COPY ./models /code/models

COPY ./tokenizer /code/tokenizer

COPY main.py /code/main.py

COPY new2oldmatch.json /code/new2oldmatch.json

COPY something.py /code/something.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]