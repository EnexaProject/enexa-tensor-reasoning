import tnreason.model.tensor_model as tm
import tnreason.contraction.core_contractor as coc

import tnreason.model.generate_test_data as gtd

expressionsDict =     {
        "e0" : [["a1","imp","a2"], 2],
        "e1" : [["a4","eq",["not","a1"]], 2],
        "e2":  [["a4", "xor", ["a5","eq", "a1"]], 2],
        "e3":  [["a6", "or", ["not", "a1"]], 2],
    }

print(gtd.generate_sampleDf(expressionsDict, 100))


tRep = tm.TensorRepresentation(expressionsDict)

contractor = coc.CoreContractor(tRep.get_cores())
contractor.optimize_coreList()
contractor.create_instructionList_from_coreList()
contractor.visualize()

contractor.contract()