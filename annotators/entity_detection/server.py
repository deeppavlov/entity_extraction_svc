import logging
import os
import re
import time

import sentry_sdk
from flask import Flask, jsonify, request

from deeppavlov import build_model

sentry_sdk.init(os.getenv("SENTRY_DSN"))

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

config_name = os.getenv("CONFIG")

try:
    entity_detection_alexa = build_model(config_name, download=True)
    entity_detection_lcquad = build_model("entity_detection_lcquad.json", download=True)
    entity_detection_alexa(["what is the capital of russia"])
    logger.info("entity detection model is loaded.")
except Exception as e:
    sentry_sdk.capture_exception(e)
    logger.exception(e)
    raise e

nltk_stopwords_file = "nltk_stopwords.txt"
nltk_stopwords = [line.strip() for line in open(nltk_stopwords_file, "r").readlines()]
EVERYTHING_EXCEPT_LETTERS_DIGITALS_AND_SPACE = re.compile(r"[^a-zA-Z0-9 \-&*+]")
DOUBLE_SPACES = re.compile(r"\s+")


def get_result(request, prev_utt=False):
    st_time = time.time()
    last_utterances = request.json.get("last_utterances", [])
    prev_utterances = request.json.get("prev_utterances", [])
    logger.info(f"input (the last utterances): {last_utterances} (prev utterances): {prev_utterances}")

    utterances_list = []
    utterances_nums = []
    if prev_utt:
        for n, (prev_utt, last_utt) in enumerate(zip(prev_utterances, last_utterances)):
            total_utt = " ".join(prev_utt) + " " + " ".join(last_utt)
            total_utt = total_utt.replace("  ", " ").strip()
            utterances_list.append(total_utt)
            utterances_nums.append(n)
        logger.info(f"utterances concat {utterances_list}")
    else:
        for n, utterances in enumerate(last_utterances):
            for elem in utterances:
                if len(elem) > 0:
                    if elem[-1] not in {".", "!", "?"}:
                        elem = f"{elem}."
                    utterances_list.append(elem.lower())
                    utterances_nums.append(n)

    utt_entities_batch = [{} for _ in last_utterances]
    utt_entities = {}
    if utterances_list:
        entities_batch, tags_batch, positions_batch, entities_offsets_batch, probas_batch = entity_detection_alexa(
            utterances_list
        )
        (
            entities_batch_lc,
            tags_batch_lc,
            positions_batch_lc,
            entities_offsets_batch_lc,
            probas_batch_lc,
        ) = entity_detection_lcquad(utterances_list)
        logger.info(f"entities_batch_lcquad {entities_batch_lc}")
        already_detected_set = set()
        cur_num = 0
        for entities_list, tags_list, entities_offsets_list, num in zip(
            entities_batch, tags_batch, entities_offsets_batch, utterances_nums
        ):
            if num != cur_num:
                utt_entities_batch[cur_num] = utt_entities
                utt_entities = {}
                cur_num = num
            for entity, tag, offsets in zip(entities_list, tags_list, entities_offsets_list):
                if entity not in nltk_stopwords and len(entity) > 2:
                    entity = EVERYTHING_EXCEPT_LETTERS_DIGITALS_AND_SPACE.sub(" ", entity)
                    entity = DOUBLE_SPACES.sub(" ", entity).strip()
                    if "entities" in utt_entities:
                        utt_entities["entities"].append(entity)
                        utt_entities["labelled_entities"].append(
                            {"text": entity, "label": tag.lower(), "offsets": offsets}
                        )
                        already_detected_set.add((entity, offsets))
                    else:
                        utt_entities["entities"] = [entity]
                        utt_entities["labelled_entities"] = [{"text": entity, "label": tag.lower(), "offsets": offsets}]
                        already_detected_set.add((entity, offsets))
        cur_num = 0
        for entities_list, tags_list, entities_offsets_list, num in zip(
            entities_batch_lc, tags_batch_lc, entities_offsets_batch_lc, utterances_nums
        ):
            if num != cur_num:
                utt_entities_batch[cur_num] = utt_entities
                utt_entities = {}
                cur_num = num
            for entity, tag, offsets in zip(entities_list, tags_list, entities_offsets_list):
                if entity not in nltk_stopwords and len(entity) > 2:
                    entity = EVERYTHING_EXCEPT_LETTERS_DIGITALS_AND_SPACE.sub(" ", entity)
                    entity = DOUBLE_SPACES.sub(" ", entity).strip()
                    found_already_detected = False
                    for already_detected_entity, already_detected_offsets in already_detected_set:
                        if entity == already_detected_entity or (
                            offsets[0] >= already_detected_offsets[0] and offsets[1] <= already_detected_offsets[1]
                        ):
                            found_already_detected = True
                    if "entities" in utt_entities:
                        if not found_already_detected:
                            utt_entities["entities"].append(entity)
                            utt_entities["labelled_entities"].append(
                                {"text": entity, "label": "misc", "offsets": offsets}
                            )
                    else:
                        if not found_already_detected:
                            utt_entities["entities"] = [entity]
                            utt_entities["labelled_entities"] = [{"text": entity, "label": "misc", "offsets": offsets}]
        filtered_utt_entities = {}
        filtered_entities = []
        filtered_labelled_entities = []
        filtered_out_numbers = []
        for n1, (entity1, labelled_entity1) in enumerate(
            zip(utt_entities.get("entities", []), utt_entities.get("labelled_entities", []))
        ):
            st1, end1 = labelled_entity1["offsets"]
            for n2, (entity2, labelled_entity2) in enumerate(
                zip(utt_entities.get("entities", []), utt_entities.get("labelled_entities", []))
            ):
                st2, end2 = labelled_entity2["offsets"]
                if n1 != n2 and entity1 != entity2 and st2 >= st1 and end2 <= end1:
                    filtered_out_numbers.append(n2)
        for n1, (entity1, labelled_entity1) in enumerate(
            zip(utt_entities.get("entities", []), utt_entities.get("labelled_entities", []))
        ):
            if n1 not in filtered_out_numbers:
                filtered_entities.append(entity1)
                filtered_labelled_entities.append(labelled_entity1)
        filtered_utt_entities["entities"] = filtered_entities
        filtered_utt_entities["labelled_entities"] = filtered_labelled_entities

        if utt_entities:
            utt_entities_batch[cur_num] = filtered_utt_entities

    if not last_utterances:
        utt_entities_batch.append({})

    total_time = time.time() - st_time
    logger.info(f"entity detection exec time: {total_time: .3f}s")
    logger.info(f"entity_detection, input {last_utterances}, output {utt_entities_batch}")
    return utt_entities_batch


@app.route("/respond", methods=["POST"])
def respond():
    result = get_result(request)
    if result == [{}]:
        result = get_result(request, prev_utt=True)

    # postprocess
    return jsonify(result)


@app.route("/respond_batch", methods=["POST"])
def respond_batch():
    result = get_result(request)
    return jsonify([{"batch": result}])


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=9103)
