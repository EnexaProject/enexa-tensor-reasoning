def contract(coreDict, openColors, method="PgmpyVariableEliminator"):
    if method == "NumpyEinsum":
        from tnreason.contraction.numpy_contractor import NumpyEinsumContractor
        contractor = NumpyEinsumContractor(coreDict=coreDict, openColors=openColors)
        return contractor.contract()
    elif method == "PgmpyVariableEliminator":
        from tnreason.contraction.pgmpy_contractor import PgmpyVariableEliminator
        contractor = PgmpyVariableEliminator(coreDict=coreDict, openColors=openColors)
        return contractor.contract()

    else:
        raise ValueError("Contractor Type {} not supported.".format(method))

def get_core(coreType="NumpyTensorCore"):
    if coreType == "NumpyTensorCore":
        from tnreason.engine.cores import NumpyTensorCore
        return NumpyTensorCore
    else:
        raise ValueError("Core Type {} not supported.".format(coreType))