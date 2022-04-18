FROM python:3.7.6

RUN apt-get -y update && \
    apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

RUN apt-get install -y sqlite3

ARG NER_CONFIG
ARG EL_CONFIG
ARG PORT
ARG SRC_DIR
ARG SED_ARG=" | "

ENV NER_CONFIG=$NER_CONFIG
ENV EL_CONFIG=$EL_CONFIG
ENV PORT=$PORT

RUN pip install pybind11==2.2.4
RUN pip install hdt==2.3

COPY ./annotators/entity_extraction/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY $SRC_DIR /src

WORKDIR /src
RUN python -m deeppavlov install $NER_CONFIG
RUN python -m deeppavlov install $EL_CONFIG

CMD python server.py
