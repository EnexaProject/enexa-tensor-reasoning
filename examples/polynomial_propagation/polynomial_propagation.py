from tnreason import engine
import numpy as np
from queue import Queue


class PolynomialPropagator:

    def __init__(self, coresDict):
        self.coresDict = coresDict
        self.find_colors()
        self.create_affectionDict()

        self.coreQueue = Queue()

    def find_colors(self):
        self.colors = []
        for coreKey in self.coresDict:
            for color in self.coresDict[coreKey].colors:
                if color not in self.colors:
                    self.colors.append(color)

    def create_affectionDict(self):
        self.affectionDict = {coreKey: [] for coreKey in self.coresDict}
        for sendCore in self.coresDict:
            for receiveCore in self.coresDict:
                for color in self.coresDict[sendCore].colors:
                    if color in self.coresDict[receiveCore].colors and sendCore != receiveCore:
                        self.affectionDict[sendCore].append([receiveCore, color])

    def propagate_cores(self, coreKeys=None):
        if coreKeys is None:
            coreKeys = list(self.coresDict.keys())
        for sendCore in coreKeys:
            for receiveCore, color in self.affectionDict[sendCore]:
                self.coreQueue.put([sendCore, receiveCore, color])

        while not self.coreQueue.empty():
            self.propagation_step(self.coreQueue.get())

    def propagation_step(self, tripleSpec):
        sendingKey, receivingKey, color = tripleSpec
        changed = propagate_fromToAlong(self.coresDict[sendingKey], self.coresDict[receivingKey], color)
        if changed:
            for receiveCore, color in self.affectionDict[sendingKey]:
                self.coreQueue.put([sendingKey, receiveCore, color])


def propagate_fromToAlong(sendingCore, receivingCore, color):
    ## Colors in sendingCore
    changed = False
    if all([color in posDict for scalar, posDict in sendingCore.values.slices]):
        message = np.unique([posDict[color] for scalar, posDict in sendingCore.values.slices])

        newSlices = []
        for scalar, posDict in receivingCore.values.slices:
            if color in posDict:
                if posDict[color] in message:
                    newSlices.append((scalar, posDict))
                else:
                    changed = True
            else:
                newSlices.append((scalar, posDict))
        receivingCore.values = engine.SliceValues(newSlices,
                                                  receivingCore.values.shape)
    return changed


if __name__ == "__main__":
    core1 = engine.get_core("PolynomialCore")(
        values=engine.SliceValues([(1, {"a": 3, "b": 2}), (2, {"a": 2, "b": 3})],
                                  shape=[5, 4]),
        colors=["a", "b"]
    )
    core2 = engine.get_core("PolynomialCore")(
        values=engine.SliceValues([(1, {"a": 3, "c": 2}), (2, {"a": 0, "c": 3})],  # (1, {"c":1}),
                                  shape=[5, 4]),
        colors=["a", "c"]
    )
    print("### Test1")
    pp = PolynomialPropagator({"c1": core1, "c2": core2})
    pp.propagate_cores()
    for core in pp.coresDict:
        print(pp.coresDict[core])

    core3 = engine.get_core("PolynomialCore")(
        values=engine.SliceValues([(1, {"b": 3, "a": 2, "d": 2}), (2, {"b": 0, "a": 3})],
                                  shape=[5, 4]),
        colors=["b", "d", "e", "a"]
    )

    print("### Test2")
    pp2 = PolynomialPropagator({"c1": core1, "c2": core2, "c3": core3})
    pp2.propagate_cores()
    for core in pp.coresDict:
        print(pp.coresDict[core])