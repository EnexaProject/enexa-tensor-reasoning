from tnreason import engine

core = engine.create_tensor_encoding(
    inshape=[2, 2],
    incolors=["a", "b"],
    function=lambda i, j: i + j,
    coreType="TentrisCore"
)

print([val for val in core.values])

core = engine.create_relational_encoding(
    inshape=[2, 2],
    outshape=[2],
    incolors=["a", "b"],
    outcolors=["c"],
    function=lambda i, j: [i + j],
    coreType="TentrisCore"
)

print([val for val in core.values])

from tnreason import encoding

coreDict = encoding.create_raw_formula_cores(expression=["imp", "a", "b"], coreType="TentrisCore")
print(coreDict)

print([value for value in coreDict["(imp_a_b)_conCore"].values])


