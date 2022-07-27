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
