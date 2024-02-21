from tnreason.tensor import formula_tensors as ft

from tnreason.contraction import core_contractor as coc

expression = ["A1", "and", ["not", ["A2","and",["not","A1"]]]]
fTensor = ft.FormulaTensor(expression, "f1", headType="expFactor", weight=1)


contractor = coc.CoreContractor(fTensor.get_cores())
contractor.optimize_coreList()
contractor.create_instructionList_from_coreList()
contractor.visualize()


#cv.draw_contractionDiagram(fTensor.get_cores(), title="FormulaTensor")
#cv.draw_contractionDiagram({**fTensor.subExpressionCoresDict, "head": fTensor.headCore},title="FormulaTensor")