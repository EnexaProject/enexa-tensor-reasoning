from tnreason import encoding
from tnreason.encoding.formulas import headCoreSuffix

from tnreason import algorithms
from tnreason.algorithms.constraint_propagation import domainCoreSuffix


class KnowledgePropagator:
    def __init__(self, knowledgeBase, evidenceDict={}):
        self.atoms = knowledgeBase.atoms
        print(self.atoms)
        self.knowledgeCores = {
            **encoding.create_formulas_cores({**knowledgeBase.weightedFormulasDict, **knowledgeBase.factsDict}),
            **encoding.create_constraints(knowledgeBase.categoricalConstraintsDict),
            **encoding.create_evidence_cores(evidenceDict)}
        self.knownHeads = get_evidence_headKeys(evidenceDict) + [
            encoding.get_formula_color(knowledgeBase.factsDict[key]) + headCoreSuffix for key in
            knowledgeBase.factsDict]

    def evaluate(self):
        propagator = algorithms.ConstraintPropagator(binaryCoresDict=self.knowledgeCores)
        propagator.initialize_domainCoresDict()
        propagator.propagate_cores(coreOrder=self.knownHeads)

        self.entailedDict = propagator.find_assignments()

    def find_carrying_cores(self):
        return {key: self.knowledgeCores[key] for key in self.knowledgeCores if not
        all([color in self.entailedDict for color in self.knowledgeCores[key].colors]) and all([
            color not in self.atoms for color in self.knowledgeCores[key].colors
        ])}


def get_evidence_headKeys(evidenceDict):
    return [encoding.get_formula_color(key) + headCoreSuffix for key in evidenceDict if
            evidenceDict[key]] + [
        encoding.get_formula_color(["not", key]) + headCoreSuffix for key in evidenceDict if
        not evidenceDict[key]
    ]
