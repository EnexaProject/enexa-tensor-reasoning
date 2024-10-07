from tnreason import engine, encoding


class EnergyGibbs:
    def __init__(self, energyDict, affectionDict=None, colors=[], dimDict={}):
        self.energyDict = energyDict
        self.colors = colors

        if affectionDict is None:
            self.affectionDict = {color: list(self.energyDict.keys()) for color in colors}
        else:
            self.affectionDict = affectionDict

        self.dimDict = dimDict
        self.sample = {}

    def initialize_sample_uniform(self):
        for color in self.colors:
            self.sample.update(
                encoding.create_trivial_core(color + "_probCore", self.dimDict[color], [color]).draw_sample())

    def calculate_energy(self, upColors):
        affectedEnergyKeys = list(set().union(*[self.affectionDict[color] for color in upColors]))
        sampleCores = {
            color + "_sampleCore": encoding.create_basis_core(color + "_sampleCore", [self.dimDict[color]], [color],
                                                              (self.sample[color])) for
            color in self.sample if color not in upColors}
        contractedEnergy = engine.contract(coreDict={**self.energyDict[affectedEnergyKeys[0]][0], **sampleCores},
                                           openColors=upColors, dimDict=self.dimDict).multiply(
            self.energyDict[affectedEnergyKeys[0]][1])
        for energyKey in affectedEnergyKeys[1:]:
            contractedEnergy = contractedEnergy.sum_with(
                engine.contract({**self.energyDict[energyKey][0], **sampleCores}, openColors=upColors,
                                dimDict=self.dimDict).multiply(
                    self.energyDict[energyKey][1]))
        return contractedEnergy

    def sample_colors(self, colors, temperature=1):
        energy = self.calculate_energy(colors)
        self.sample.update(energy.draw_sample(asEnergy=True, temperature=temperature))


if __name__ == "__main__":
    energyDict = {
        "w1": [{**encoding.create_raw_formula_cores(["imp", "a", "b"]),
                **encoding.create_head_core(["imp", "a", "b"], headType="truthEvaluation")}, -1],
        "w2": [{**encoding.create_raw_formula_cores(["xor", "b", "c"]),
                **encoding.create_head_core(["xor", "b", "c"], headType="truthEvaluation")}, 1]
    }

    sampler = EnergyGibbs(energyDict, colors=["a", "b", "c"], dimDict={"a": 2, "b": 2, "c": 2})
    sampler.initialize_sample_uniform()

    sampler.sample_colors(["a", "b"], temperature=10)
    sampler.sample_colors(["b", "c"], temperature=10)
    print(sampler.sample)