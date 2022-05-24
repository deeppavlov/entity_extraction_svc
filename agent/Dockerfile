FROM python:3.9

ARG PORT
ARG SRC_DIR

ENV PORT=$PORT

COPY ./agent/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY $SRC_DIR /src
WORKDIR /src

CMD sh -c 'bin/wait && uvicorn server:app --host 0.0.0.0 --port $PORT --reload'
