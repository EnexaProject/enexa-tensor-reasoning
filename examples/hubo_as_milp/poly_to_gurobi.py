import gurobipy as gp
from gurobipy import GRB


def poly_to_gurobi_model(polyCore):
    model = gp.Model(str(polyCore.name)+"_gurobiModel")

    variableDict = {
        color: model.addVar(vtype=GRB.BINARY, name=color) for color in polyCore.colors
    }

    slackVariableDict = {
        "slack" + str(j): model.addVar(vtype=GRB.BINARY, name="slack" + str(j)) for j in range(len(polyCore.values))
    }

    for j, entry in enumerate(polyCore.values):
        lowBound = 1
        for var in entry[1]:
            if entry[1][var] == 1:
                lowBound = lowBound + variableDict[var] - 1
                model.addConstr(slackVariableDict["slack" + str(j)] <= variableDict[var])
            elif entry[1][var] == 0:
                lowBound = lowBound - variableDict[var]
                model.addConstr(slackVariableDict["slack" + str(j)] <= (1 - variableDict[var]))
            else:
                raise ValueError("Index {} not supported, binary only!".format(entry[1][var]))
        model.addConstr(lowBound <= slackVariableDict["slack" + str(j)])

    objective = 0
    for j, entry in enumerate(polyCore.values):
        objective = objective + entry[0] * slackVariableDict["slack" + str(j)]

    model.setObjective(objective, GRB.MAXIMIZE)
    return model


if __name__ == "__main__":
    from examples.cnf_representation import formula_to_polynomial_core as ftp

    polyCore = ftp.weightedFormulas_to_polynomialCore({
        "w1": ["imp", "a", "b", 0.678],
        "w2": ["a", 0.34]
    })
    print(polyCore)

    model = poly_to_gurobi_model(polyCore)
    model.optimize()

    # Output the results
    for v in model.getVars():
        print(f'{v.varName}: {v.x}')