import requests
from datetime import datetime
from deeppavlov_kg.core import ontology
import deeppavlov_kg.core.graph as graph


entities = [
    {
        "immutable": {
            "Id": "1",
        },
        "mutable": {
            "name": "Dragon",
            "kind": "CustomKinds.Manufacturing.Space.Spacecraft"
        }
    },
    {
        "immutable": {
            "Id":"2",
        },
        "mutable": {
            "name": "SpaceX",
            "kind": "CustomKinds.Organizations.Company"
        }
    },
    {
        "immutable": {
            "Id":"3",
        },
        "mutable": {
            "name": "NASA",
            "kind": "ZU.Kinds.Organization"
        }
    }
]

edges = [
    (
        "2",
        "CustomRelations.Business.IsParnerOf",
        {"on": datetime.strptime("2022-01-20", "%Y-%m-%d")},
        "3"
    )
]

kind = "entity"

for entity_dict in entities:
    ontology.create_kind(kind, kind_properties=set(entity_dict["mutable"].keys()))
    graph.create_entity(kind, entity_dict["immutable"]["Id"], entity_dict["mutable"])

for id_a, rel, rel_dict, id_b in edges:
    graph.create_relationship(id_a, rel, rel_dict,id_b)

res = graph.search_nodes()
print(res)

EL_URL = "http://0.0.0.0:9103"

res = requests.post(f"{EL_URL}/parse_custom_kg", json={})
print("res", res)

text = "SpaceX just set a new record for its fastest Dragon astronaut trip yet. Elon Musk spaceflight company launched four Crew-4 astronauts to the International Space Station for NASA in less than 16 hours on Wednesday (April 27), the shortest flight time since SpaceX began crewed flights in 2020."

res = requests.post(f"{EL_URL}/entity_extraction", json={"texts": [text]})
resp = res.json()
for key in resp:
    print(key, resp[key])
