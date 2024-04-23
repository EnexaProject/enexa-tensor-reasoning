from tnreason import encoding
from tnreason.encoding.formulas import headCoreSuffix

from tnreason import algorithms


class KnowledgePropagator:
    def __init__(self, knowledgeBase, evidenceDict={}):
        self.atoms = knowledgeBase.atoms
        self.knowledgeCores = {
            **encoding.create_formulas_cores({**knowledgeBase.weightedFormulasDict, **knowledgeBase.factsDict}),
            **encoding.create_constraints(knowledgeBase.categoricalConstraintsDict),
            **encoding.create_evidence_cores(evidenceDict)}

        self.propagator = algorithms.ConstraintPropagator(binaryCoresDict=self.knowledgeCores)

        self.knownHeads = get_evidence_headKeys(evidenceDict) + [
            encoding.get_formula_color(knowledgeBase.factsDict[key]) + headCoreSuffix for key in
            knowledgeBase.factsDict]

    def evaluate(self, variables=None):
        if variables is None:
            variables = self.knownHeads
        self.propagator.initialize_domainCoresDict()
        self.propagator.propagate_cores(coreOrder=variables)
        self.entailedDict = self.propagator.find_assignments()
        return self.entailedDict

    def find_carrying_cores(self, variables=None, variablesShape={}):
        if variables is None:
            variables = self.atoms
        return self.propagator.find_variable_cone(variables, {**variablesShape,
                                                              **{variable: 2 for variable in variables if
                                                                 variable not in variablesShape}})


def get_evidence_headKeys(evidenceDict):
    return [encoding.get_formula_color(key) + headCoreSuffix for key in evidenceDict if
            evidenceDict[key]] + [
        encoding.get_formula_color(["not", key]) + headCoreSuffix for key in evidenceDict if
        not evidenceDict[key]
    ]
