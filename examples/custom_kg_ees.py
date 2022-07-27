import requests


EL_URL = "http://0.0.0.0:9103"

res = requests.post(f"{EL_URL}/parse_custom_kg", json={})
print("res", res)

text = "SpaceX just set a new record for its fastest Dragon astronaut trip yet. Elon Musk spaceflight company launched four Crew-4 astronauts to the International Space Station for NASA in less than 16 hours on Wednesday (April 27), the shortest flight time since SpaceX began crewed flights in 2020."

res = requests.post(f"{EL_URL}/entity_extraction", json={"texts": [text]})
resp = res.json()
for key in resp:
    print(key, resp[key])
