from tnreason import encoding
from tnreason import algorithms


class KnowledgePropagator:
    """
    Evaluates formulas by constraint propagation.
    """
    def __init__(self, knowledgeBase, evidenceDict={}):
        self.atoms = knowledgeBase.atoms
        self.knowledgeCores = {
            **knowledgeBase.create_cores(),
            **encoding.create_evidence_cores(evidenceDict)}

        self.propagator = algorithms.ConstraintPropagator(binaryCoresDict=self.knowledgeCores)

        self.knownHeads = get_evidence_headKeys(evidenceDict) + [
            encoding.get_formula_color(knowledgeBase.facts[key]) + encoding.headCoreSuffix for key in
            knowledgeBase.facts]

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
    return [encoding.get_formula_color(key) + encoding.headCoreSuffix for key in evidenceDict if
            evidenceDict[key]] + [
        encoding.get_formula_color(["not", key]) + encoding.headCoreSuffix for key in evidenceDict if
        not evidenceDict[key]
    ]
