from tnreason.encoding import storage
from tnreason import encoding

loaded = storage.load_from_yaml("./model.yaml")
print(encoding.get_knowledge_cores(loaded).keys())

