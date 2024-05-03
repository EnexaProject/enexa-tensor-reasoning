from tnreason import encoding
from tnreason import engine

probFormulasKey = "weightedFormulas"
logFormulasKey = "facts"
categoricalsKey = "categoricalConstraints"
evidenceKey = "evidence"


class EmpiricalDistribution:
    def __init__(self, sampleDf, atomKeys=None):
        if atomKeys is None:
            atomKeys = list(sampleDf.columns)
        self.atoms = atomKeys
        self.sampleDf = sampleDf
        self.dataNum = sampleDf.values.shape[0]

    def create_cores(self):
        return encoding.create_data_cores(self.sampleDf, self.atoms)

    def get_partition_function(self, allAtoms=[]):
        unseenAtomNum = len([atom for atom in allAtoms if atom not in self.atoms])
        return (self.dataNum * (2 ** unseenAtomNum))


class HybridKnowledgeBase:
    def __init__(self, weightedFormulas={}, facts={}, categoricalConstraints={}, evidence={}):
        self.weightedFormulas = weightedFormulas
        self.facts = facts
        self.categoricalConstraints = categoricalConstraints
        self.evidence = evidence

        self.find_atoms()

    def find_atoms(self):
        self.atoms = encoding.get_all_atoms(
            {**{key: self.weightedFormulas[key][:-1] for key in self.weightedFormulas},
             **self.facts})
        for constraintKey in self.categoricalConstraints:
            for atom in self.categoricalConstraints[constraintKey]:
                if atom not in self.atoms:
                    self.atoms.append(atom)
        for eKey in self.evidence:
            if eKey not in self.atoms:
                self.atoms.append(eKey)
        self.atoms = list(self.atoms)

    def from_yaml(self, loadPath):
        modelSpec = encoding.load_from_yaml(loadPath)
        if probFormulasKey in modelSpec:
            self.weightedFormulas = modelSpec[probFormulasKey]
        if logFormulasKey in modelSpec:
            self.facts = modelSpec[logFormulasKey]
        if categoricalsKey in modelSpec:
            self.categoricalConstraints = modelSpec[categoricalsKey]
        if evidenceKey in modelSpec:
            self.evidence = modelSpec[evidenceKey]
        self.find_atoms()

    def to_yaml(self, savePath):
        encoding.storage.save_as_yaml({
            probFormulasKey: self.weightedFormulas,
            logFormulasKey: self.facts,
            categoricalsKey: self.categoricalConstraints,
            evidenceKey: self.evidence
        }, savePath)

    def include(self, secondHybridKB):
        self.weightedFormulas = {**self.weightedFormulas,
                                 **secondHybridKB.weightedFormulas}
        self.facts = {**self.facts,
                      **secondHybridKB.facts}
        self.categoricalConstraints = {**self.categoricalConstraints,
                                       **secondHybridKB.categoricalConstraints}
        self.evidence = {**self.evidence,
                         **secondHybridKB.evidence}
        self.find_atoms()

    def create_cores(self):
        return {**encoding.create_formulas_cores({**self.weightedFormulas, **self.facts}),
                **encoding.create_evidence_cores(self.evidence),
                **encoding.create_constraints(self.categoricalConstraints)}

    def get_partition_function(self, allAtoms=[]):
        unseenAtomNum = len([atom for atom in allAtoms if atom not in self.atoms])
        return (engine.contract(coreDict=self.create_cores(), openColors=[]).values
                * (2 ** unseenAtomNum))

    def is_satisfiable(self):
        return engine.contract(coreDict={**encoding.create_formulas_cores(self.facts),
                                         **encoding.create_evidence_cores(self.evidence),
                                         **encoding.create_constraints(self.categoricalConstraints)},
                               openColors=[]).values > 0