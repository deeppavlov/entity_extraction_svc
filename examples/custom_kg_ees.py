import requests


EL_URL = "http://0.0.0.0:9103"
path = f"/root/.deeppavlov/downloads/deeppavlov_kg/database"

db_url = "bolt://neo4j:neo4j@10.11.1.102:7687"
ontology_kinds_hierarchy_path = f"{path}/ontology_kinds_hierarchy.pickle"
ontology_data_model_path = f"{path}/ontology_data_model.json"
db_ids_file_path = f"{path}/db_ids.txt"

res = requests.post(f"{EL_URL}/parse_custom_kg", json={"db_url": db_url,
                                                       "ontology_kinds_hierarchy_path": ontology_kinds_hierarchy_path,
                                                       "ontology_data_model_path": ontology_data_model_path,
                                                       "db_ids_file_path": db_ids_file_path})
print("res", res)

text = "SpaceX just set a new record for its fastest Dragon astronaut trip yet. Elon Musk spaceflight company launched four Crew-4 astronauts to the International Space Station for NASA in less than 16 hours on Wednesday (April 27), the shortest flight time since SpaceX began crewed flights in 2020."

res = requests.post(f"{EL_URL}/entity_extraction", json={"texts": [text]})
resp = res.json()
for key in resp:
    print(key, resp[key])
