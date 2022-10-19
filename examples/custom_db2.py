import os
import requests
from pathlib import Path
from deeppavlov.core.data.utils import simple_download


custom_db = ['<C1> <Kind> "CustomKinds.Manufacturing.Space.Spacecraft" .',
             '<C1> <Title> "Dragon" .',
             '<C1> <CustomRelations.Manufacturing.IsBuiltBy> <C2> .',
             '<C2> <Kind> "CustomKinds.Organizations.Company" .',
             '<C2> <Title> "SpaceX" .',
             '<C2> <CustomRelations.Business.IsParnerOf> <C3> .',
             '<C3> <Kind> "ZU.Kinds.Organization" .',
             '<C3> <Title> "NASA" .']

EL_URL = "http://0.0.0.0:9103"

res = requests.post(f"{EL_URL}/kb_schema", json={"relation_info": {"label_rel": "Title", "type_rel": "Kind"}})
print("res", res)

res = requests.post(f"{EL_URL}/add_kb", json={"triplets": custom_db})
print("res", res)

text = "SpaceX just set a new record for its fastest Dragon astronaut trip yet. Elon Musk spaceflight company launched four Crew-4 astronauts to the International Space Station for NASA in less than 16 hours on Wednesday (April 27), the shortest flight time since SpaceX began crewed flights in 2020."

res = requests.post(f"{EL_URL}/entity_extraction", json={"texts": [text]})
resp = res.json()
for key in resp:
    print(key, resp[key])
