# knowledge/__init__.py

from tnreason.knowledge.deductive import HybridInferer
from tnreason.knowledge.weight_estimation import EmpiricalDistribution, EntropyMaximizer
from tnreason.knowledge.formula_boosting import FormulaBooster
from tnreason.knowledge.batch_evaluation import KnowledgePropagator

from tnreason import encoding

probFormulasKey = "weightedFormulas"
logFormulasKey = "facts"
categoricalsKey = "categoricalConstraints"
evidenceKey = "evidence"


def load_kb_from_yaml(loadPath):
    kb = HybridKnowledgeBase()
    kb.from_yaml(loadPath)
    return kb


class HybridKnowledgeBase:
    def __init__(self, weightedFormulas={}, facts={}, categoricalConstraints={}, evidence={}):
        self.weightedFormulas = weightedFormulas
        self.facts = facts
        self.categoricalConstraints = categoricalConstraints
        self.evidence = evidence

        self.find_atoms()

    def find_atoms(self):
        self.atoms = encoding.get_all_variables({**self.weightedFormulas, **self.facts})
        for constraintKey in self.categoricalConstraints:
            for atom in self.categoricalConstraints[constraintKey]:
                if atom not in self.atoms:
                    self.atoms.append(atom)
        for eKey in self.evidence:
            if eKey not in self.atoms:
                self.atoms.append(eKey)

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

    def to_yaml(self, savePath):
        encoding.storage.save_as_yaml({
            probFormulasKey: self.weightedFormulas,
            logFormulasKey: self.facts,
            categoricalsKey: self.categoricalConstraints,
            evidenceKey: self.evidence
        }, savePath)

    def include(self, secondHybridKB):
        self.weightedFormulas = {**self.weightedFormulas,
                                 **secondHybridKB.weightedFormulasDict}
        self.facts = {**self.facts,
                      **secondHybridKB.factsDict}
        self.categoricalConstraints = {**self.categoricalConstraints,
                                       **secondHybridKB.categoricalConstraints}
        self.evidence = {**self.evidence,
                         **secondHybridKB.evidence}
        self.find_atoms()

    def create_cores(self, hardOnly=False):
        if hardOnly:
            return {**encoding.create_formulas_cores({**self.weightedFormulas, **self.facts}),
                    **encoding.create_evidence_cores(self.evidence),
                    **encoding.create_constraints(self.categoricalConstraints)}
        else:
            return {**encoding.create_formulas_cores(self.facts),
                    **encoding.create_evidence_cores(self.evidence),
                    **encoding.create_constraints(self.categoricalConstraints)}
