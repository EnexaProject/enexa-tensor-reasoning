from tnreason.model import tensor_model as tm
from tnreason.model import formula_tensors as ft

from tnreason.contraction import core_contractor as coc


class TensorKB:
    def __init__(self, formulaList):
        self.formulaList = formulaList
        self.formulaTensors = tm.TensorRepresentation(
            {"f" + str(i): [formula, 1] for i, formula in enumerate(formulaList)}, headType="truthEvaluation")

        print("Initialized a Knowledge Base with the formulas: \n  {}".format(
            "\n  ".join([str(f) for f in self.formulaList])))

    def tell(self, formula):
        assert self.formulaTensors.headType == "truthEvaluation"

        self.formulaList.append(formula)
        self.formulaTensors.add_expression(formula, formulaKey="f" + str(len(self.formulaList)))

        print("{} has been added to the Knowledge Base.".format(formula))

    def ask(self, formula):
        if coc.CoreContractor({**self.formulaTensors.get_cores(headType="truthEvaluation"),
                               **ft.FormulaTensor(expression=["not", formula],
                                                  headType="truthEvaluation").get_cores()
                               }).contract().values == 0:
            print("{} is entailed by the Knowledge Base.".format(formula))
            return "entailed"
        elif coc.CoreContractor({**self.formulaTensors.get_cores(headType="truthEvaluation"),
                                 **ft.FormulaTensor(expression=formula,
                                                    headType="truthEvaluation").get_cores()
                                 }).contract().values == 0:
            print("{} is contradicted by the Knowledge Base.".format(formula))
            return "contradicting"
        else:
            print("{} is neither entailed nor contradicted by the Knowledge Base.".format(formula))
            return "contingent"
