from tnreason.engine.engine_visualization import draw_contractionDiagram

defaultCoreType="NumpyTensorCore"
defaultContractionMethod = "PgmpyVariableEliminator"

def contract(coreDict, openColors, method=defaultContractionMethod, outPut=defaultCoreType):
    if len(coreDict)==0:
        from tnreason.engine.numpy_contractor import EmptyCore
        return EmptyCore()
    if method == "NumpyEinsum":
        from tnreason.engine.numpy_contractor import NumpyEinsumContractor
        return NumpyEinsumContractor(coreDict=coreDict, openColors=openColors).contract()
    elif method == "TensorFlowEinsum":
        from tnreason.engine.tensorflow_contractor import TensorFlowContractor
        if outPut == "NumpyTensorCore":
            return TensorFlowContractor(coreDict=coreDict, openColors=openColors).einsum().to_NumpyTensorCore()
    elif method == "TorchEinsum":
        from tnreason.engine.torch_contractor import TorchContractor
        if outPut == "NumpyTensorCore":
            return TorchContractor(coreDict=coreDict, openColors=openColors).einsum().to_NumpyTensorCore()

    elif method == "PgmpyVariableEliminator":
        from tnreason.engine.pgmpy_contractor import PgmpyVariableEliminator
        return PgmpyVariableEliminator(coreDict=coreDict, openColors=openColors).contract()

    else:
        raise ValueError("Contractor Type {} not supported with Output {}.".format(method, outPut))


def get_core(coreType="NumpyTensorCore"):
    if coreType == "NumpyTensorCore":
        from tnreason.engine.numpy_contractor import NumpyCore
        return NumpyCore
    else:
        raise ValueError("Core Type {} not supported.".format(coreType))
