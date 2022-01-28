import json
import re
from requests import Session
from nltk import sent_tokenize
from dandelion import extract_dandelion


# download test set from http://dl.fbaipublicfiles.com/KILT/wned-dev-kilt.jsonl

with open("wned-dev-kilt.jsonl", 'r') as fl:
    lines  = fl.readlines()

correct = 0
total = 0

with Session() as sess:
    for n, line in enumerate(lines[:1000]):
        sample = json.loads(line.strip())
        text = sample["input"]
        fnd = re.findall(r"\[START_ENT\] (.*) \[END_ENT\]", text)
        mention = fnd[0]
        gold_page = sample["output"][0]["provenance"][0]["title"]
        
        text = text.replace("[START_ENT]", "").replace("[END_ENT]", "").replace("  ", " ")
        sentences = sent_tokenize(text)
        sentence_with_entity = ""
        for sentence in sentences:
            if mention in sentence:
                sentence_with_entity = sentence
                break
        
        if sentence_with_entity:
            data = extract_dandelion(sess, sentence_with_entity)
            try:
                found = False
                for elem in data["annotations"]:
                    if elem["spot"] == mention and gold_page.lower() == elem["title"].lower():
                        found = True
                        break
                if found:
                    correct += 1
                total += 1
            except:
                pass
        if total == 500:
            break

print("accuracy", correct / total)
