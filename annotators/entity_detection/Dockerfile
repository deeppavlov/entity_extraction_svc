FROM deeppavlov/base-gpu:0.12.1

ARG SEQ_TAG_CONFIG
ARG CONFIG
ARG LOWERCASE
ARG PORT
ARG SRC_DIR
ARG SED_ARG=" | "

ENV SEQ_TAG_CONFIG=$SEQ_TAG_CONFIG
ENV CONFIG=$CONFIG
ENV LOWERCASE=$LOWERCASE
ENV PORT=$PORT

COPY ./annotators/entity_detection/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY $SRC_DIR /src

WORKDIR /src
RUN python -m deeppavlov install $SEQ_TAG_CONFIG

RUN sed -i "s|$SED_ARG|g" "$SEQ_TAG_CONFIG"

CMD gunicorn  --workers=1 --timeout 500 server:app -b 0.0.0.0:9103
