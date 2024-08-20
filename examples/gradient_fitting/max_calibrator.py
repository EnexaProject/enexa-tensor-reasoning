from tnreason import engine, encoding
import numpy as np

class MaxCalibrator:
    def __init__(self, clusters):
        self.clusters = clusters  # double dict: clusterKey, then tensor network
        self.clusterConnectivity = {}

    def compute_max_message(self, sendClusterKey, sendColors, useMessages={}):
        rawCore = engine.contract({**self.clusters[sendClusterKey], **useMessages},
                                  openColors=get_all_colors(self.clusters[sendClusterKey]))
        return engine.get_core()(
            values=np.max(rawCore.values,
                          axis=tuple([i for (i, color) in enumerate(rawCore.colors) if color not in sendColors])),
            colors=sendColors)

    def max_propagation(self, sendList):
        self.messages = {}
        for sendClusterKey, receiveClusterKey in sendList:
            sendColors = list(set(get_all_colors(self.clusters[sendClusterKey])) & set(
                get_all_colors(self.clusters[receiveClusterKey])))
            self.messages[sendClusterKey + "_" + receiveClusterKey] = self.compute_max_message(
                sendClusterKey, sendColors,
                useMessages={key: self.messages[key] for key in self.messages if key.split("_")[
                    1] == sendClusterKey})

    def get_max_assignment(self, clusterList):
        self.max_assignment = {}
        colorDimDict = {}

        for clusterKey in clusterList:
            clusterColors = get_all_colors(self.clusters[clusterKey])
            rawCore = engine.contract({**self.clusters[clusterKey],
                                       **{key: self.messages[key] for key in self.messages if key.split("_")[
                                           1] == clusterKey},
                                       **{color + "_knownMax": encoding.create_basis_core(
                                           color + "_knownMax", shape=[colorDimDict[color]], colors=[color],
                                           numberTuple=(self.max_assignment[color]))
                                           for color in clusterColors if color in self.max_assignment}},
                                      openColors=clusterColors)
            maxAssignment = np.unravel_index(np.argmax(rawCore.values), rawCore.values.shape)
            for i, color in enumerate(rawCore.colors):
                if color not in self.max_assignment:
                    self.max_assignment[color] = maxAssignment[i]
                    colorDimDict[color] = rawCore.values.shape[i]


def get_all_colors(network):
    colors = []
    for core in network:
        for color in network[core].colors:
            if color not in colors:
                colors.append(color)
    return colors


if __name__ == "__main__":
    import numpy as np

    from tnreason import encoding

    cluster0 = {"fun": encoding.create_random_core(name="fun", shape=(2, 10, 2), colors=["a", "b", "c"]),
                "fun2": encoding.create_random_core(name="fun2", shape=(2), colors=["a"])}

    cluster1 = {"fun": encoding.create_random_core(name="fun", shape=(2, 10, 2), colors=["d", "b", "e"]),
                "fun2": encoding.create_random_core(name="fun2", shape=(10), colors=["b"])}

    maksik = MaxCalibrator(clusters={"c0": cluster0, "c1": cluster1})
    maksik.compute_max_message("c0", ["b"])

    maksik.max_propagation(sendList=[("c0", "c1"), ("c1", "c0")])
    maksik.get_max_assignment(["c1", "c0"])

    print(maksik.max_assignment)
