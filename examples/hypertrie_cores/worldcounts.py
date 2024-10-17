from tnreason import engine, encoding

#formula = ["imp","a",["and","b","c"]]
formula = ["imp","a","b"]

cores = encoding.create_raw_formula_cores(formula, coreType="HypertrieCore")

contracted = engine.contract(coreDict=cores, openColors=[encoding.get_formula_color(formula)],method="TentrisEinsum")

print([entry for entry in contracted.values])
