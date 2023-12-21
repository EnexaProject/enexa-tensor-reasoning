from tnreason.contraction import contraction_generation as cg
from tnreason.contraction import core_contractor as coc

from tnreason.logic import basis_calculus as bc

test_expression = [["not","P1"],"and",["not","P2"]]
#test_expression = ["P1","and","P2"]

print([str(test_expression)+str(test_expression)])
formulaDict = cg.generate_factor_dict(test_expression,headType="truthEvaluation")


def inspect_formulaDict(insFormulaDict):
    for key in insFormulaDict:
        print("###")
        print(key)
        print(insFormulaDict[key].colors)
        print(insFormulaDict[key].values)

#inspect_formulaDict(formulaDict)

contractor = coc.CoreContractor(formulaDict, openColors=["P1","P2"])
contractor.optimize_coreList()
contractor.create_instructionList_from_coreList()
print(contractor.instructionList)

fullcore = contractor.contract()
print(fullcore.colors)
print(fullcore.values.shape)

print(fullcore.values)