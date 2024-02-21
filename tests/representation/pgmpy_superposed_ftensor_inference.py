from tnreason.tensor import superposed_formula_tensors as sft

from tnreason.representation import pgmpy_inference as pinf

spfTensor = sft.SuperPosedFormulaTensor(["jaszczur", "piskle", ["sledz", "sikorka"]],
                                    {
                                        "piskle": ["and", "or", "imp"],
                                        # "not1" : ["not"],
                                        # "sledz": ["alphasledz"],
                                        "sikorka": ["sikorka12"],
                                        "sledz": ["not"],
                                        "jaszczur": ["a1", "a2", "a3"]
                                    },
                                        coreType="CoordinateCore")

cores = spfTensor.get_cores()
cores = spfTensor.get_cores()
for key in cores:
    print(key, cores[key].values.shape, cores[key].colors)

inferer = pinf.PgmpyInferer(spfTensor.get_cores())
print(inferer.query(["jaszczur"], {"a1":1}))



