from tnreason.encoding.formulas import create_formulas_cores, create_raw_formula_cores, get_formula_color, \
    create_headCore, create_evidence_cores
from tnreason.encoding.constraints import create_constraints
from tnreason.encoding.auxiliary import create_emptyCoresDict, get_variables, get_all_variables
from tnreason.encoding.storage import save_as_yaml, load_from_yaml
from tnreason.encoding.neurons import create_neuron, create_architecture, find_atoms, find_selection_dimDict, \
    create_solution_expression
from tnreason.encoding.data import create_data_cores