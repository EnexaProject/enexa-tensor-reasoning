def get_contractor(contractionMethod="TNChainContractor"):
    if contractionMethod == "TNChainContractor":
        from tnreason.contraction.core_contractor import CoreContractor
        return CoreContractor
    elif contractionMethod == "PgmpyVariableEliminator":
        from tnreason.contraction.pgmpy_contractor import PgmpyVariableEliminator
        return PgmpyVariableEliminator
    elif contractionMethod == "NumpyEinsum":
        from tnreason.contraction.numpy_contractor import NumpyEinsumContractor
        return NumpyEinsumContractor
    else:
        raise ValueError("Contractor Type {} not supported.".format(contractionMethod))