from tnreason import engine
from tnreason import encoding
from tnreason import network

from tnreason.knowledge import logic_model as lm
from tnreason.knowledge import knowledge_visualization as knv

defaultContractionMethod = "PgmpyVariableEliminator"


def from_yaml(loadPath):
    modelSpec = encoding.load_from_yaml(loadPath)

    if "weightedFormulas" in modelSpec:
        weightedFormulas = modelSpec["weightedFormulas"]
    else:
        weightedFormulas = {}

    if "facts" in modelSpec:
        facts = modelSpec["facts"]
    else:
        facts = {}

    if "categoricalConstraints" in modelSpec:
        categoricalConstraints = modelSpec["categoricalConstraints"]
    else:
        categoricalConstraints = {}

    return HybridKnowledgeBase(weightedFormulasDict=weightedFormulas,
                               factsDict=facts,
                               categoricalConstraintsDict=categoricalConstraints)


class HybridKnowledgeBase:
    def __init__(self, weightedFormulasDict={}, factsDict={}, categoricalConstraintsDict={}):
        self.weightedFormulasDict = {key: [weightedFormulasDict[key][0], float(weightedFormulasDict[key][1])]
                                     for key in weightedFormulasDict}
        self.factsDict = factsDict.copy()
        self.categoricalConstraintsDict = categoricalConstraintsDict.copy()

        self.atoms = list(
            encoding.get_all_variables([weightedFormulasDict[key][0] for key in weightedFormulasDict] +
                                       [factsDict[key] for key in factsDict]))
        if not len(self.factsDict) == 0:
            if not self.is_satisfiable():
                raise ValueError("The initialized Knowledge Base is inconsistent!")

    def create_cores(self):
        return {**encoding.create_formulas_cores({**self.weightedFormulasDict, **self.factsDict}),
                **encoding.get_constraint_cores(self.categoricalConstraintsDict)}

    def include(self, secondHybridKB):

        if not len(self.factsDict) == 0:
            if not self.is_satisfiable():
                raise ValueError("By including additional facts, the Knowledge Base got inconsistent!")

        self.weightedFormulasDict = {**self.weightedFormulasDict,
                                     **secondHybridKB.weightedFormulasDict}
        self.factsDict = {**self.factsDict,
                          **secondHybridKB.factsDict}
        self.atoms = list(set(self.atoms) | set(secondHybridKB.atoms))

    def is_satisfiable(self, contractionMethod=defaultContractionMethod):
        return engine.contract(method=contractionMethod, coreDict={**encoding.create_formulas_cores(self.factsDict),
                                                                   **encoding.get_constraint_cores(
                                                                       self.categoricalConstraintsDict)},
                               openColors=[]).values > 0

    def ask_constraint(self, constraint):
        probability = self.ask(constraint, evidenceDict={})
        if probability > 0.9999:
            return "entailed"
        elif probability == 0:
            return "contradicting"
        else:
            return "contingent"

    def tell_constraint(self, constraint, constraintKey=None):
        if constraintKey is None:
            constraintKey = "c" + str(len(self.factsDict))
        answer = self.ask_constraint(constraint)
        if answer == "entailed":
            print("{} is redundant to the Knowledge Base and has not been added.".format(constraint))
            return "not added"
        elif answer == "contradicting":
            print("{} would make the Knowledge Base inconsistent and has not been added.".format(constraint))
            return "not added"
        else:
            self.factsDict[constraintKey] = constraint

    def tell(self, formula, weight, formulaKey=None):
        if formulaKey is None:
            formulaKey = "f" + str(len(self.weightedFormulasDict))

        self.weightedFormulasDict[formulaKey] = [formula, weight]

        for atom in encoding.get_variables(formula):
            if atom not in self.atoms:
                self.atoms.append(atom)

    def ask(self, queryFormula, evidenceDict={}, contractionMethod=defaultContractionMethod):

        contracted = engine.contract(
            coreDict={**encoding.create_formulas_cores(
                {**self.weightedFormulasDict, **self.factsDict, **evidence_to_expressionsDict(evidenceDict)}),
                      **encoding.get_constraint_cores(self.categoricalConstraintsDict),
                      **encoding.create_raw_formula_cores(queryFormula)
                      },
            method=contractionMethod, openColors=[encoding.get_formula_color(queryFormula)]).values

        return contracted[1] / (contracted[0] + contracted[1])

    def query(self, variableList, evidenceDict={}, contractionMethod=defaultContractionMethod):
        return engine.contract(method=contractionMethod, coreDict={
            **encoding.create_emptyCoresDict([variable for variable in variableList if
                                              variable not in self.atoms and variable not in evidenceDict]),
            **encoding.create_formulas_cores(
                {**self.weightedFormulasDict, **self.factsDict,
                 **evidence_to_expressionsDict(evidenceDict)}),
            **encoding.get_constraint_cores(self.categoricalConstraintsDict)
        }, openColors=variableList).normalize()

    def exact_map_query(self, variableList, evidenceDict={}):
        distributionCore = self.query(variableList, evidenceDict)
        maxIndex = distributionCore.get_maximal_index()
        return {variable: maxIndex[i] for i, variable in enumerate(distributionCore.colors)}

    def annealed_map_query(self, variableList, evidenceDict={}, annealingPattern=[(10, 1), (5, 0.1), (2, 0.01)]):
        ## Need to support heating in distributions first!
        return self.gibbs_sample(variableList, evidenceDict)

    def gibbs_sample(self, variableList, evidenceDict={}, sweepNum=10):
        logRep = lm.LogicRepresentation(self.weightedFormulasDict, self.factsDict)
        logRep.infer(evidenceDict=evidenceDict, simplify=True)
        weightedFormulas, facts = logRep.get_formulas_and_facts()

        distribution = network.TNDistribution({**encoding.create_formulas_cores({**weightedFormulas, **facts}),
                                               **encoding.get_constraint_cores(self.categoricalConstraintsDict)})

        return distribution.gibbs_sampling(variableList, {variable: 2 for variable in variableList}, sweepNum=sweepNum)

    def evaluate_evidence(self, evidenceDict={}):
        return lm.LogicRepresentation(self.weightedFormulasDict, self.factsDict).evaluate_evidence(evidenceDict)

    def to_yaml(self, savePath):
        encoding.storage.save_as_yaml({
            "weightedFormulas": self.weightedFormulasDict,
            "facts": self.factsDict,
            "categoricalConstraints": self.categoricalConstraintsDict
        }, savePath)

    def visualize(self, evidenceDict={}, strengthMultiplier=4, strengthCutoff=10, fontsize=10, showFormula=True,
                  pos=None):
        return knv.visualize_knowledge(self.weightedFormulasDict,
                                       factsDict=self.factsDict,
                                       strengthMultiplier=strengthMultiplier,
                                       strengthCutoff=strengthCutoff,
                                       fontsize=fontsize,
                                       showFormula=showFormula,
                                       evidenceDict=evidenceDict,
                                       pos=pos)


def evidence_to_expressionsDict(evidenceDict):
    return {**{key: key for key in evidenceDict if evidenceDict[key]},
            **{key: ["not", key] for key in evidenceDict if not evidenceDict[key]}
            }
