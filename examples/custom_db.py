import os
import requests
from pathlib import Path
from deeppavlov.core.data.utils import simple_download


save_path = "~/.deeppavlov/downloads/entity_linking_eng"
db_filename = "custom_db.nt"
db_path = Path(os.path.join(save_path, db_filename)).expanduser().resolve()
db_path.parent.mkdir(parents=True, exist_ok=True)

url = "http://files.deeppavlov.ai/deeppavlov_data/entity_linking/custom_db.nt"
simple_download(url, db_path)


EL_URL = "http://0.0.0.0:9103"

res = requests.post(f"{EL_URL}/kb_schema", json={"relation_info": {"label_rel": "label", "type_rel": "type"}})
print("res", res)

fl = open(str(db_path), 'r')
lines = fl.readlines()

res = requests.post(f"{EL_URL}/add_kb", json={"triplets": lines})
print("res", res)

text = "SpaceX just set a new record for its fastest Dragon astronaut trip yet. Elon Musk spaceflight company launched four Crew-4 astronauts to the International Space Station for NASA in less than 16 hours on Wednesday (April 27), the shortest flight time since SpaceX began crewed flights in 2020."

res = requests.post(f"{EL_URL}/entity_extraction", json={"texts": [text]})
resp = res.json()
for key in resp:
    print(key, resp[key])
