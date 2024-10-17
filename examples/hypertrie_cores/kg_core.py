from tnreason.engine import workload_to_tentris as wt

kgCore = wt.ht_from_rdf("/home/examples/hypertrie_cores/THWS_demo.ttl")
print([val for val in kgCore.values])

print(kgCore.get_shape())

print(kgCore.values[1407374883553337, 1407374883553338, 1697645953286153])