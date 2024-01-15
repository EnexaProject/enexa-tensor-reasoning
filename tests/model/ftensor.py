from tnreason.model import formula_tensors as ft

from tnreason.contraction import contraction_visualization as cv

expression = ["A1", "and", ["not", ["A2","and","A1"]]]
fTensor = ft.FormulaTensor(expression, "f1", headType="expFactor", weight=1)

cv.draw_contractionDiagram({**fTensor.subExpressionCoresDict, "head": fTensor.headCore},title="FormulaTensor")