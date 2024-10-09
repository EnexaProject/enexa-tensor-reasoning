defaultContractionMethod = "PgmpyVariableEliminator"

def contract(coreDict, openColors, dimDict={}, method=defaultContractionMethod):
    """
    Contractors are initialized with
        * coreDict: Dictionary of colored tensor cores specifying a network
        * openColors: List of colors to leave open in the contraction
        * dimDict: Dictionary of dimension to each color, required only when colors do not appear in the cores
    """

    from tnreason.engine.workload_to_numpy import NumpyCore
    appearingColors = list(set().union(*[coreDict[coreKey].colors for coreKey in coreDict]))
    for color in openColors:
        if color not in appearingColors:
            if color not in dimDict:
                dimDict[color] = 2 # Default variable size used, when not specified otherwise
            coreDict[color + "_trivialCore"] = NumpyCore(values=[1 for i in range(dimDict[color])], colors=[color],
                                                         name=color + "_trivialCore")
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