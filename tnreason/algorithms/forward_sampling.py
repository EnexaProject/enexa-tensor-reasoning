from tnreason import engine


class ForwardSampler:
    def __init__(self, networkCores, dimDict=None, coreType=None, contractionMethod=None):
        self.networkCores = networkCores
        if dimDict is None:
            self.dimDict = engine.get_dimDict(networkCores)
        else:
            self.dimDict = dimDict
        self.coreType = coreType
        self.contractionMethod = contractionMethod

    def draw_forward_sample(self, colors):
        for color in colors:
            if color not in self.dimDict:
                self.dimDict[color] = 2
        sample = {}
        for sampleColor in colors:
            condProb = engine.contract(
                {**self.networkCores,
                 **{oldColor + "_sampleCore": engine.create_basis_core(oldColor + "_sampleCore",
                                                                       [self.dimDict[oldColor]], [oldColor],
                                                                       sample[oldColor], coreType=self.coreType) for
                    oldColor in sample}},
                openColors=[sampleColor], dimDict=self.dimDict, method=self.contractionMethod)
            sample[sampleColor] = condProb.draw_sample(asEnergy=False)[sampleColor]
        return sample
