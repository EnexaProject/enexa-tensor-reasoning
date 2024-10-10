from examples.rencoding.generate_rencoding import generate_relational_encoding, create_coreDict_relational_encoding

from tnreason import knowledge


def samples_to_map(samples, variableList):
    return lambda k: samples[variableList].iloc[k].values


if __name__ == "__main__":
    dataNum = 10

    samples = knowledge.InferenceProvider(
        knowledge.HybridKnowledgeBase(weightedFormulas={"w": ["imp", "a", "b", 1]})).draw_samples(dataNum)

    # As a single core
    dataCore = generate_relational_encoding(inshape=[dataNum], outshape=[2, 2], incolors=["j"], outcolors=["a", "b"],
                                            function=samples_to_map(samples, ["a", "b"]))

    # As a tensor network given the partition structure
    dataCores = create_coreDict_relational_encoding(inshape=[dataNum], outshape=[2, 2], incolors=["j"], outcolors=["a", "b"],
                                            function=samples_to_map(samples, ["a", "b"]),
                                                    partitionDict={"fun":["a","b"]})

    import numpy as np
    print(np.linalg.norm(dataCore.values - dataCores["fun_encodingCore"].values))