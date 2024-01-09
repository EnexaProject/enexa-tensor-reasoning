from tnreason.model import formula_tensors as ft
import tnreason.model.generate_test_data as gtd
from tnreason.logic import coordinate_calculus as cc

from tnreason.contraction import contraction_visualization as cv

import numpy as np

learnedFormulaDict = {
        "f0": ["A1", 10],
        "f1": [["not", ["A2", "and", "A3"]], 5],
        "f2": ["A2", 2]
    }

sampleDf = gtd.generate_sampleDf(learnedFormulaDict, 10)
print(sampleDf)

skeletonExpression = ["P1","and",["not","P2"]]
candidatesDict = {"P1": ["A1", "A2"], "P2": ["A2"]}
parameterCoresDict = {
        "vCore1": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P1", "H1"]),
        "vCore2": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P2", "H1"]),
    }

supFtensor = ft.SuperposedFormulaTensor(skeletonExpression, candidatesDict, parameterCoresDict)
supFtensor.create_atomDataCores(sampleDf)


cv.draw_contractionDiagram({**supFtensor.parameterCoresDict,
                                **supFtensor.skeletonCoresDict,
                                **supFtensor.selectorCoresDict,
                                **supFtensor.dataCoresDict})