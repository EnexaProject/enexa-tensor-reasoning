def architecture_from_skeleton(skeletonExpression, candidatesDict, neuronName):
    architecture = {neuronName: get_neuron(skeletonExpression, candidatesDict, neuronName)}
    subCounter = 0
    for subSkeleton in skeletonExpression[1:]:
        if not isinstance(subSkeleton, str):
            architecture = {**architecture,
                            **architecture_from_skeleton(subSkeleton, candidatesDict,
                                                         neuronName + "." + str(subCounter))}
            subCounter += 1
    return architecture


def get_neuron(skeletonExpression, candidatesDict, neuronName):
    if skeletonExpression[0] in candidatesDict:  ## First position is a connective placeholder
        connectives = candidatesDict[skeletonExpression[0]]
    else:  ## First position is a connective
        connectives = [skeletonExpression[0]]

    specList = [connectives]
    subCounter = 0
    for subExpression in skeletonExpression[1:]:
        if isinstance(subExpression, str):
            if subExpression in candidatesDict:
                specList.append(candidatesDict[subExpression])
            else:
                specList.append([subExpression])
        else:
            specList.append([neuronName + "." + str(subCounter)])
            subCounter += 1
    return specList


if __name__ == "__main__":
    print(
        architecture_from_skeleton(
            ["and", ["c1", "p1"], ["c2", "p2", "p3"]],
            {"p1": ["a", "b"],
             "p2": ["blub"]},
            "headNeuron"
        )
    )
