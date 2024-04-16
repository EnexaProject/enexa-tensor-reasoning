from tnreason.encoding import storage
from tnreason.encoding import neurons

spec = {
    "funneur" : {
        "connectiveList" : ["imp","or"],
        "candidatesList" : [
            ["a1","a2"],
            ["a3", "a1"]
        ]
    },
    "funneur2": {
        "connectiveList": ["imp", "or"],
        "candidatesList": [
            ["funneur", "a2"],
            ["funneur", "a1"]
        ]
    }
}

print(neurons.create_architecture(spec).keys())

storage.save_as_yaml(spec, "./funneur.yaml")
loaded = storage.load_from_yaml("./funneur.yaml")
print(loaded)
print(neurons.create_architecture(loaded).keys())

from tnreason import  encoding

print(encoding.create_architecture(loaded))
print(encoding.load_architecture_cores("./funneur.yaml"))

#print(get_neuron_cores(
#    "funneur", connectiveList=["not"],
#    candidatesDict={"sledz": ["jaszczur", "sikorka"]}
#))
