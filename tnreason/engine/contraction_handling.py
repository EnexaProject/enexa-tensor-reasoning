defaultContractionMethod = "NumpyEinsum"


def contract(coreDict, openColors, dimDict={}, method=None):
    """
    Contractors are initialized with
        * coreDict: Dictionary of colored tensor cores specifying a network
        * openColors: List of colors to leave open in the contraction
        * dimDict: Dictionary of dimension to each color, required only when colors do not appear in the cores
    """
    if method is None:
        method = defaultContractionMethod

    ## Handling trivial colors (not appearing in coreDict)
    from tnreason.engine.auxiliary_cores import create_trivial_core
    dimDict.update({color : 2 for color in openColors if color not in dimDict})
    if len(coreDict) == 0:
        return create_trivial_core(name="Contracted", shape=[dimDict[color] for color in openColors], colors=openColors)
    appearingColors = list(set().union(*[coreDict[coreKey].colors for coreKey in coreDict]))
    for color in openColors:
        if color not in appearingColors:
            coreDict[color + "_trivialCore"] = create_trivial_core(color + "_trivialCore", shape=[dimDict[color]],
                                                                   colors=[color])

    ## Einstein Summation Contractors
    if method == "NumpyEinsum":
        from tnreason.engine.workload_to_numpy import NumpyEinsumContractor
        return NumpyEinsumContractor(coreDict=coreDict, openColors=openColors).contract()
    elif method == "TensorFlowEinsum":
        from tnreason.engine.workload_to_tensorflow import TensorFlowContractor
        return TensorFlowContractor(coreDict=coreDict, openColors=openColors).einsum().to_NumpyTensorCore()
    elif method == "TorchEinsum":
        from tnreason.engine.workload_to_torch import TorchContractor
        return TorchContractor(coreDict=coreDict, openColors=openColors).einsum().to_NumpyTensorCore()
    elif method == "TentrisEinsum":
        from tnreason.engine.workload_to_tentris import TentrisContractor
        return TentrisContractor(coreDict=coreDict, openColors=openColors).einsum().to_NumpyTensorCore()

    ## Variable Elimination Contractors
    elif method == "PgmpyVariableEliminator":
        from tnreason.engine.workload_to_pgmpy import PgmpyVariableEliminator
        return PgmpyVariableEliminator(coreDict=coreDict, openColors=openColors).contract()

    ## Experimental Polynomial Contraction
    elif method == "PolynomialContractor":
        from tnreason.engine.polynomial_contractor import GenericSliceContractor
        return GenericSliceContractor(coreDict=coreDict, openColors=openColors).contract()


    else:
        raise ValueError("Contractor Type {} not known.".format(method))
