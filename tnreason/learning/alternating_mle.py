import tnreason.logic.expression_utils as eu

import tnreason.representation.sampledf_to_cores as stoc

import tnreason.contraction.bc_contraction_generation as bcg
import tnreason.contraction.expression_evaluation as ee
import tnreason.contraction.core_contractor as coc
import tnreason.contraction.layered_core_contractor as lcoc

import tnreason.logic.coordinate_calculus as cc

import numpy as np


class MLEOptimizerBase:
    def __init__(self, skeletonExpression, candidatesDict, variableCoresDict, learnedFormulaDict={}):
        self.skeleton = skeletonExpression  ## Can just handle pure conjunctions!
        self.skeletonPlaceholders = eu.get_variables(skeletonExpression)

        self.candidatesDict = candidatesDict

        self.create_atom_selectors()  ## Creates self.atomSelectorDict

        self.variableCoresDict = variableCoresDict
        self.learnedFormulaDict = learnedFormulaDict

        self.candidatesAtomsList = []
        for candidatesKey in candidatesDict:
            for atom in candidatesDict[candidatesKey]:
                if atom not in self.candidatesAtomsList:
                    self.candidatesAtomsList.append(atom)

    def create_atom_selectors(self):
        ## Creates atom selector cores selecting atom activation
        self.atomSelectorDict = {}
        for variableKey in self.skeletonPlaceholders:
            atoms = self.candidatesDict[variableKey]
            shape = [len(atoms)] + [2 for atom in atoms]
            values = np.zeros(shape=shape)  ## Bottleneck! Can instead do Hadamards with selection matrices.
            for i, atom in enumerate(atoms):
                pos = [i] + [0 for pos in range(i)] + [1] + [0 for pos in range(len(atoms) - i - 1)]
                tuplePos = tuple(entry for entry in pos)
                values[tuplePos] = 1
            self.atomSelectorDict["Selector_" + variableKey] = cc.CoordinateCore(values, [variableKey, *atoms])

    def create_mln_core(self):
        ## Creates exponentiated factors of other formulas
        self.formulaCoreDict = bcg.generate_formulaCoreDict(self.learnedFormulaDict)

    def create_exponentiated_variables(self):
        ## Creates exponentiated factor to the variables tbo
        ## To do: Think of removing this bottleneck: Constract the current variables to a single core instead of exp TN!
        ## Missing for generality (not just pure conjunctions): Skeleton in form of another contraction dict
        variablesContractor = coc.CoreContractor({**self.variableCoresDict, **self.atomSelectorDict},
                                                 openColors=self.candidatesAtomsList)
        variablesContractor.optimize_coreList()
        variablesContractor.create_instructionList_from_coreList()
        self.variablesExpFactor = variablesContractor.contract().exponentiate()

    def create_variable_gradient(self, tboVariableKey):
        return {key: self.variableCoresDict[key] for key in self.variableCoresDict if key != tboVariableKey}

    def contract_partition_gradient(self, tboVariableKey):
        contractionDict = {**self.formulaCoreDict, "variablesExpFactor": self.variablesExpFactor,
                           **self.create_variable_gradient(tboVariableKey),
                           **self.atomSelectorDict}
        contractor = coc.CoreContractor(contractionDict, openColors=self.variableCoresDict[tboVariableKey].colors)
        contractor.optimize_coreList()
        contractor.create_instructionList_from_coreList()
        return contractor.contract().multiply(1 / self.compute_partition())

    def compute_likelihood(self):
        ## Without learned formulas!
        contractor = coc.CoreContractor(coreDict={**self.fixedCoresDict, **self.variableCoresDict})
        contractor.optimize_coreList()
        contractor.create_instructionList_from_coreList()
        return contractor.contract().values / (self.dataNum * self.compute_partition())

    def compute_partition(self):
        contractionDict = {"variablesExpFactor": self.variablesExpFactor,
                           **self.formulaCoreDict}
        contractor = coc.CoreContractor(coreDict=contractionDict, openColors=[])
        contractor.optimize_coreList()
        contractor.create_instructionList_from_coreList()
        return contractor.contract().values

    ## Data Side: Compute Gradient
    def create_fixedCores(self, sampleDf):
        self.dataNum = sampleDf.values.shape[0]
        self.fixedCoresDict = {}
        # Supports only and connectivity!
        for atomKey in self.skeletonPlaceholders:
            self.fixedCoresDict[atomKey] = stoc.create_fixedCore(sampleDf, self.candidatesDict[atomKey], ["j", atomKey],
                                                                 atomKey)

    def contract_data_gradient(self, tboVariableKey):
        ## Contract the variable gradient with the fixedCores
        contractor = coc.CoreContractor(
            coreDict={**self.fixedCoresDict, **self.create_variable_gradient(tboVariableKey)},
            openColors=self.variableCoresDict[tboVariableKey].colors)
        contractor.optimize_coreList()
        contractor.create_instructionList_from_coreList()
        return contractor.contract()

    ## Optimization preparation
    def random_initialize_variableCoresDict(self):
        for coreKey in self.variableCoresDict:
            self.variableCoresDict[coreKey].values = np.random.random(size=self.variableCoresDict[coreKey].values.shape)


class GradientDescentMLE(MLEOptimizerBase):

    def compute_gradient(self, tboVariableKey):
        partition_gradient = self.contract_partition_gradient(tboVariableKey)
        data_gradient = self.contract_data_gradient(tboVariableKey)
        return partition_gradient.sum_with(data_gradient.multiply(-1))

    def descent_step(self, tboVariableKey, stepWidth):
        self.variableCoresDict[tboVariableKey] = self.variableCoresDict[tboVariableKey].sum_with(
            self.compute_gradient(tboVariableKey).multiply(-1 * stepWidth)
        )

    def alternating_gradient_descent(self, sweepNum, stepWidth, computeLikelihood=True, verbose=True):
        likelihoods = np.empty(shape=(sweepNum, len(self.variableCoresDict)))
        for sweep in range(sweepNum):
            for i, variableKey in enumerate(self.variableCoresDict):
                self.descent_step(variableKey, stepWidth)
                if computeLikelihood:
                    likelihoods[sweep, i] = self.compute_likelihood()
        return likelihoods


class AlternatingNewtonMLE(MLEOptimizerBase):

    def contract_gradient_exponential(self, tboVariableKey):
        contractionDict = {**self.formulaCoreDict, "variablesExpFactor": self.variablesExpFactor,
                           **self.create_variable_gradient(tboVariableKey),
                           **self.atomSelectorDict}
        contractor = coc.CoreContractor(contractionDict, openColors=self.variableCoresDict[tboVariableKey].colors)
        contractor.optimize_coreList()
        contractor.create_instructionList_from_coreList()
        return contractor.contract()

    def contract_double_gradient_exponential(self, tboVariableKey):
        contractionDict = {**self.formulaCoreDict, "variablesExpFactor": self.variablesExpFactor,
                           **self.create_variable_gradient(tboVariableKey),
                           **self.atomSelectorDict,
                           **copy_CoreDict(self.create_variable_gradient(tboVariableKey), suffix="out"),
                           **copy_CoreDict(self.atomSelectorDict, suffix="out",
                                           exceptionColors=self.candidatesAtomsList),
                           }
        openColors = self.variableCoresDict[tboVariableKey].colors
        openColors = openColors + [color + "_out" for color in openColors]
        contractor = coc.CoreContractor(contractionDict, openColors=openColors)
        contractor.optimize_coreList()
        contractor.create_instructionList_from_coreList()
        return contractor.contract()

    ## Algorithm
    def alternating_newton(self, sweepNum=10, dampFactor=0.001, monotoneousCondition=True, regParameter=0,
                           computeLikelihood=True, verbose=True):
        self.likelihood = self.compute_likelihood()
        likelihoods = np.empty(shape=(sweepNum, len(self.variableCoresDict)))
        for sweep in range(sweepNum):
            print("### SWEEP {} ###".format(sweep))
            for i, variableKey in enumerate(self.variableCoresDict):
                self.newton_step(variableKey, dampFactor=dampFactor, monotoneusCondition=monotoneousCondition,
                                 regParameter=regParameter, verbose=verbose)
                likelihoods[sweep, i] = self.likelihood
        return likelihoods

    def newton_step(self, tboVariableKey, dampFactor, monotoneusCondition, regParameter, verbose=True):
        expGradient = self.contract_gradient_exponential(tboVariableKey)

        vector = expGradient.sum_with(
            self.contract_data_gradient(tboVariableKey).multiply(-1 / (self.dataNum * self.compute_partition())))

        ## Operator
        doubleExpGradient = self.contract_double_gradient_exponential(tboVariableKey).multiply(-1)

        empOperator = expGradient.clone()
        empOperator.colors = [color + "_out" for color in empOperator.colors]
        empOperator = empOperator.compute_and(self.contract_data_gradient(tboVariableKey).multiply(1 / self.dataNum))

        operator = empOperator.sum_with(doubleExpGradient)

        ## Flatten
        outColors = [color for color in operator.colors if color.endswith("_out")]
        inColors = [color for color in operator.colors if color not in outColors]

        outshape = [operator.values.shape[i] for i, color in enumerate(operator.colors) if color in outColors]
        inshape = [operator.values.shape[i] for i, color in enumerate(operator.colors) if color in inColors]

        outDim = np.prod(outshape)
        inDim = np.prod(inshape)

        operator.reorder_colors(inColors + outColors)
        vector.reorder_colors(inColors)

        flattenedOperatorValues = operator.values.reshape(inDim, outDim) + regParameter * np.eye(inDim)
        flattenedVectorValues = vector.values.reshape(inDim)

        ## Solve Newton
        update, residuals, rank, singular_values = np.linalg.lstsq(flattenedOperatorValues, flattenedVectorValues)

        update.reshape(inshape)
        updateCore = cc.CoordinateCore(update.reshape(inshape), inColors)

        oldCore = self.variableCoresDict[tboVariableKey].clone()

        self.variableCoresDict[tboVariableKey] = self.variableCoresDict[tboVariableKey].sum_with(
            updateCore.multiply(dampFactor))

        if monotoneusCondition:
            likelihood = self.compute_likelihood()
            if likelihood < self.likelihood:
                print("Update of {} is not accepted.".format(tboVariableKey))
                self.variableCoresDict[tboVariableKey] = oldCore
            else:
                self.likelihood = likelihood

        if verbose:
            print("Update Norm", np.linalg.norm(update), "Operator Condition", np.linalg.cond(flattenedOperatorValues))
            print(self.compute_likelihood())


def copy_CoreDict(tbdDict, suffix="", exceptionColors=[]):
    doubleDict = {}
    for key in tbdDict:
        oldCore = tbdDict[key]
        newColors = []
        for color in oldCore.colors:
            if color in exceptionColors:
                newColors.append(color)
            else:
                newColors.append(color + "_" + suffix)
        doubleDict[key + "_" + suffix] = cc.CoordinateCore(oldCore.values, newColors)
    return doubleDict


if __name__ == "__main__":
    from tnreason.logic import coordinate_calculus as cc

    atomDict = {
        "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l1", "y", "z"], name="a"),
        "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l2", "q", "z"], name="b"),
        "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l3", "q", "z"], name="c"),
    }

    skeletonExpression = ["P1", "and", "P2"]
    candidatesDict = {"P1": list(atomDict.keys()),
                      "P2": list(atomDict.keys()),
                      }

    variableCoresDict = {
        "v1": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P1", "H1"]),
        "v2": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P2", "H1"]),
    }

    learnedFormulaDict = {
        "f0": ["b", 10],
        "f1": [["not", ["a", "and", "b"]], 5],
        "f2": ["c", 2]
    }

    optimizer = AlternatingNewtonMLE(skeletonExpression, candidatesDict, variableCoresDict, {})

    optimizer.random_initialize_variableCoresDict()
    optimizer.create_mln_core()

    import tnreason.model.generate_test_data as gtd

    sampleDf = gtd.generate_sampleDf(learnedFormulaDict, 100)
    optimizer.create_fixedCores(sampleDf)
    optimizer.create_exponentiated_variables()

    optimizer.alternating_newton(10)
