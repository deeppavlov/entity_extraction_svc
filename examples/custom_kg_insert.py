import os
import requests
from datetime import datetime
from deeppavlov_kg import KnowledgeGraph


username = os.getlogin()
path = f"/home/{username}/.deeppavlov/downloads/deeppavlov_kg/database"
os.makedirs(path, exist_ok=True)

graph = KnowledgeGraph(
    "bolt://neo4j:neo4j@localhost:7687",
    ontology_kinds_hierarchy_path=f"{path}/ontology_kinds_hierarchy.pickle",
    ontology_data_model_path=f"{path}/ontology_data_model.json",
    db_ids_file_path=f"{path}/db_ids.txt"
)

entities = [
    {
        "immutable": {
            "Id": "1",
        },
        "mutable": {
            "name": "Dragon",
            "type": "CustomKinds.Manufacturing.Space.Spacecraft"
        }
    },
    {
        "immutable": {
            "Id":"2",
        },
        "mutable": {
            "name": "SpaceX",
            "type": "CustomKinds.Organizations.Company"
        }
    },
    {
        "immutable": {
            "Id":"3",
        },
        "mutable": {
            "name": "NASA",
            "type": "ZU.Kinds.Organization"
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

graph.drop_database()

kind = "Entity"

graph.ontology.create_entity_kind("Entity", kind_properties={"name", "type"})
tree = graph.ontology._load_ontology_kinds_hierarchy()
if tree is None:
    print("no tree")
tree.show()
tree.show(data_property="properties")

for entity_dict in entities:
    #graph.ontology.create_entity_kind(kind, kind_properties=set(entity_dict["mutable"].keys()))
    graph.create_entity(kind, entity_dict["immutable"]["Id"], entity_dict["mutable"])

for id_a, rel, rel_dict, id_b in edges:
    graph.create_relationship(id_a, rel, rel_dict,id_b)

res = graph.search_for_entities()
print("entities", res)
