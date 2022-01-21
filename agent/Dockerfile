FROM python:3.7.6

ARG CONFIG
ARG PORT
ARG SRC_DIR

ENV CONFIG=$CONFIG
ENV PORT=$PORT

COPY ./agent/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt
COPY ./agent/server/server.py /usr/local/lib/python3.7/site-packages/deeppavlov/utils/server/server.py

COPY $SRC_DIR /src
WORKDIR /src

CMD python -m deeppavlov riseapi $CONFIG -p $PORT
