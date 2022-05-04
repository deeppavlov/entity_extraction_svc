import requests


text = "The Mona Lisa is a sixteenth century oil painting created by Leonardo. It's held at the Louvre in Paris."

res = requests.post("http://0.0.0.0:9103/entity_extraction",
                    json={"texts": [text]}).json()
print(res)
