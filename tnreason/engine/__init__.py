from tnreason.engine.engine_visualization import draw_contractionDiagram

defaultCoreType = "NumpyTensorCore"
defaultContractionMethod = "PgmpyVariableEliminator"

def contract(coreDict, openColors, method=defaultContractionMethod):
    """
    Contractors are initialized with
        * coreDict: Dictionary of colored tensor cores specifying a network
        * openColors: List of colors to leave open in the contraction
    """
    if len(coreDict) == 0:
        return EmptyCore()

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

    ## Experimental Slice Contraction
    elif method == "BinarySliceContractor":
        from tnreason.engine.binary_slice_contractor import BinarySliceContractor
        return BinarySliceContractor(coreDict=coreDict, openColors=openColors).contract()
    elif method == "GenericSliceContractor":
        from tnreason.engine.generic_slice_contractor import GenericSliceContractor
        return GenericSliceContractor(coreDict=coreDict, openColors=openColors).contract()


    else:
        raise ValueError("Contractor Type {} not known.".format(method))


def get_core(coreType="NumpyTensorCore"):
    if coreType == "NumpyTensorCore":
        from tnreason.engine.workload_to_numpy import NumpyCore
        return NumpyCore
    else:
        raise ValueError("Core Type {} not supported.".format(coreType))

class EmptyCore:
    """
    Output of an empty contraction
    """
    def __init__(self):
        self.values = 1
        self.colors = []
        self.name = "EmptyCore"