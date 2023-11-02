from logic import basis_calculus as bc
from logic import expression_calculus as ec

import numpy as np

core = bc.BasisCore(np.random.normal(size=(2,2)),["B(y)","A(z)"], "B(y)")
#print(core.values)
#core.negate()
#print(core.values)
#print(core.colors)

core1 = bc.BasisCore(np.random.normal(size=(2,2)),["A(x)","B(z)"], "B(z)")
#sum_core = core.sum_with(core1)

#print(sum_core.values)
#print(sum_core.colors)
#print(bc.create_delta_tensor(3,2).shape)

atom_dict = {
    "a" : core,
    "b" : core1
}

example_expression = [["not","a"], "and", "b"]
core = ec.calculate_core(atom_dict, example_expression)
