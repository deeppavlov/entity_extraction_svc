FROM tensorflow/tensorflow:latest-gpu

RUN apt-key del 7fa2af80  && \
    rm -f /etc/apt/sources.list.d/cuda*.list && \
    curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    -o cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get -y update
RUN apt-get install -y build-essential zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm \
    libncurses5-dev libncursesw5-dev xz-utils libffi-dev liblzma-dev

RUN apt-get -y update && \
    apt-get install -y software-properties-common && \
    apt-get update && apt-get install git -y

RUN apt-get install -y sqlite3

ARG NER_CONFIG
ARG EL_CONFIG
ARG INCLUDE_MISC
ARG MISC_THRES
ARG PORT
ARG SRC_DIR
ARG SED_ARG=" | "

ENV NER_CONFIG=$NER_CONFIG
ENV EL_CONFIG=$EL_CONFIG
ENV INCLUDE_MISC=$INCLUDE_MISC
ENV MISC_THRES=$MISC_THRES
ENV PORT=$PORT

RUN pip install Cython

COPY ./annotators/entity_extraction/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

RUN git clone https://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@github.com/deepmipt/custom_kg_svc.git
WORKDIR /custom_kg_svc
RUN pip install -e .

ENV PYTHONPATH="${PYTHONPATH}:/custom_kg_svc/deeppavlov_kg"
ENV NEO4J_USERNAME=neo4j
ENV NEO4J_PASSWORD=neo4j

COPY $SRC_DIR /src

WORKDIR /src
RUN python -m deeppavlov install $NER_CONFIG
RUN python -m deeppavlov install $EL_CONFIG

CMD python server.py
