import os
import pickle
import requests
from pathlib import Path
from deeppavlov.core.data.utils import simple_download


save_path = "~/.deeppavlov/downloads/conll2003"
data_filename = "conll2003_test.pickle"
data_path = Path(os.path.join(save_path, data_filename)).expanduser().resolve()
data_path.parent.mkdir(parents=True, exist_ok=True)

url = "http://files.deeppavlov.ai/deeppavlov_data/entity_linking/conll2003_test.pickle"
simple_download(url, data_path)

EL_URL = "http://0.0.0.0:9103"

payload={}
files=[
  ('file',('file', open(str(data_path), 'rb'), 'application/octet-stream'))
]
headers = {}

response = requests.request("POST", f"{EL_URL}/train", headers=headers, data=payload, files=files)
print(response)
