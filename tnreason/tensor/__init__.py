def get_core(coreType="NumpyTensorCore"):
    if coreType == "NumpyTensorCore":
        from tnreason.tensor.generic_cores import NumpyTensorCore
        return NumpyTensorCore
    else:
        raise ValueError("Core Type {} not supported.".format(coreType))