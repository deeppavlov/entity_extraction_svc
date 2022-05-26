FROM python:3.9

ARG PORT
ARG SRC_DIR

ENV PORT=$PORT

COPY ./agent/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY $SRC_DIR /src/$SRC_DIR
WORKDIR /src

ENV PYTHONPATH "$PYTHONPATH:/src/$SRC_DIR"
CMD sh -c 'agent/bin/wait && uvicorn agent.server:app --host 0.0.0.0 --port $PORT --reload'
