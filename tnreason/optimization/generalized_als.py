import numpy as np
import matplotlib.pyplot as plt

from tnreason.logic import expression_generation as eg, optimization_calculus as oc


class GeneralizedALS:
    def __init__(self, variableCoresDict, fixedCoresDict):
        self.variableCoresDict = variableCoresDict  # Dictionaries (!) of Coordinate Cores ## Needed to be initialized to know what legs are contracted
        self.fixedCoresDict = fixedCoresDict
        self.filterCore = None

    def set_targetCore(self, targetCore):
        self.targetCore = targetCore

    def set_filterCore(self, filterCore):
        self.filterCore = filterCore

    def sweep(self, sweepnum=1, contractionScheme=None, verbose=False):
        residua = np.empty((sweepnum, len(self.variableCoresDict)))

        if np.linalg.norm(self.targetCore.values)==0 and verbose:
            print("Warning: Target Core is Zero!")

        if contractionScheme is None:
            contractionScheme = eg.generate_conjunctions([key for key in self.variableCoresDict.keys()])

        #if verbose:
        #    for key in self.variableCoresDict:
        #        print("Before Sweeping")
        #        print(key, self.variableCoresDict[key].values)
        #res_remember = None
        for i in range(sweepnum):
            if verbose:
                print("## SWEEP {} ##".format(i))
            for k, legKey in enumerate(self.variableCoresDict):
                residua[i, k] = self.leg_optimization(legKey, contractionScheme=contractionScheme,verbose=verbose)
                if verbose:
                    print("Optimized leg {}: Residuum is {}".format(legKey, residua[i, k]))

                #assert (res_remember is None or residua[i, k]-res_remember < 0.0001 * np.linalg.norm(self.targetCore.values)), (
                #                "Residuum increased on leg {}!".format(legKey))
                #res_remember = residua[i, k]

            ## STABILIZE LEGS MAKES PROBLEMS WHEN HAVING NOTS (AFFINE LINEARITIES)!
            # self.stabilize_legs()

        if verbose:
            print("After Sweeping")
            for key in self.variableCoresDict:
                print(key,self.variableCoresDict[key].values)
        self.residua = residua

    def stabilize_legs(self):
        normproduct = 1
        for legKey in self.variableCoresDict:
            normproduct = normproduct * np.linalg.norm(self.variableCoresDict[legKey].values)
        normsoll = normproduct ** (1 / len(self.variableCoresDict))
        for legKey in self.variableCoresDict:
            self.variableCoresDict[legKey].values = self.variableCoresDict[legKey].values \
                                                    * (normsoll / np.linalg.norm(self.variableCoresDict[legKey].values))

    ## Limited to situation, where each datacore has a legcore with same key
    # contractionScheme is a nested list determining contraction to be used in a customized way, default: conjunctions
    def leg_optimization(self, legKey, contractionScheme, verbose):
        core_dict = oc.calculate_core_dict(self.variableCoresDict, self.fixedCoresDict, legKey)

        #operator, constant = ec.calculate_operator_and_constant(core_dict, contractionScheme, legKey,
        #                                                        self.variableCoresDict[legKey].colors)
        coreList = oc.calculate_coreList(core_dict, contractionScheme, self.variableCoresDict[legKey].colors)
        operator, constant = oc.operator_constant_from_coreList(coreList, self.variableCoresDict[legKey].colors)

        if self.filterCore is not None:
            operator = operator.compute_and(self.filterCore)
            constant = constant.compute_and(self.filterCore)
            if np.linalg.norm(operator.values) == 0 and verbose:
                print("Warning: Operator is Zero! Possible reasons: Zero targetCore resulted in zero solution, "
                      "filterCore is orthogonal to prefiltered operator.")

        constant = constant.negate(ignore_ones=True)
        correctedTarget = self.targetCore.sum_with(constant).compute_and(self.filterCore)

        sol = solve_lstsq(operator, correctedTarget, self.variableCoresDict[legKey].colors)
        self.variableCoresDict[legKey].values = sol

        residuum = calculate_residuum(operator, self.variableCoresDict[legKey].clone(), correctedTarget)
        return np.linalg.norm(residuum.values)

    def visualize_residua(self):
        plt.imshow(self.residua, cmap="coolwarm", vmin=0, vmax=np.linalg.norm(self.targetCore.values))

        plt.colorbar()
        plt.title("Residuum Norm Decay", fontsize=20)
        plt.xlabel("Legs")
        plt.ylabel("Iterations")

        plt.show()

def solve_lstsq(X_k_core, y_core, leg_colors):
    lhs_core = X_k_core.contract_common_colors(X_k_core, exceptions =leg_colors)
    lhs_core.reorder_colors(leg_colors + [color + "_cor" for color in leg_colors])
    variables_leg_shapes = lhs_core.values.shape[:int(len(lhs_core.values.shape)/2)]
    var_product = np.prod(variables_leg_shapes)

    lhs = lhs_core.values

    rhs_core = X_k_core.clone().contract_common_colors(y_core.clone())
    rhs = rhs_core.values

    lhs_reshaped = lhs.reshape(var_product,var_product)
    rhs_reshaped = rhs.reshape(var_product)

    solution, res, rank, s = np.linalg.lstsq(lhs_reshaped, rhs_reshaped, rcond=None)
    return solution.reshape(variables_leg_shapes)


def calculate_residuum(operator_core, solution_core, y_core):
    prediction = operator_core.clone().contract_common_colors(solution_core.clone())
    prediction = prediction.negate(ignore_ones=True)
    return prediction.sum_with(y_core.clone())


def create_core_dict(variableCoresDict, fixedCoresDict, legKey=None):
    core_dict = {}
    for coreKey in fixedCoresDict:
        if coreKey != legKey:
            core_dict[coreKey] = fixedCoresDict[coreKey].clone().contract_common_colors(
                variableCoresDict[coreKey].clone())
        else:
            core_dict[coreKey] = fixedCoresDict[coreKey].clone()
    return core_dict
