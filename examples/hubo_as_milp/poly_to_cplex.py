from docplex.mp.model import Model


def poly_to_cplex_model(polyCore):
    model = Model(str(polyCore.name) + "_gurobiModel")

    variableDict = {
        color: model.binary_var(name=color) for color in polyCore.colors
    }

    slackVariableDict = {
        "slack" + str(j): model.binary_var(name="slack" + str(j)) for j in range(len(polyCore.values))
    }

    for j, entry in enumerate(polyCore.values):
        lowBound = 1
        for var in entry[1]:
            if entry[1][var] == 1:
                lowBound = lowBound + variableDict[var] - 1
                model.add_constraint(slackVariableDict["slack" + str(j)] <= variableDict[var])
            elif entry[1][var] == 0:
                lowBound = lowBound - variableDict[var]
                model.add_constraint(slackVariableDict["slack" + str(j)] <= (1 - variableDict[var]))
            else:
                raise ValueError("Index {} not supported, binary only!".format(entry[1][var]))
        model.add_constraint(lowBound <= slackVariableDict["slack" + str(j)])

    objective = 0
    for j, entry in enumerate(polyCore.values):
        objective = objective + entry[0] * slackVariableDict["slack" + str(j)]

    model.maximize(objective)
    return model


if __name__ == "__main__":
    from examples.cnf_representation import formula_to_polynomial_core as ftp

    polyCore = ftp.weightedFormulas_to_polynomialCore({
        "w1": ["imp", "a", "b", 0.678],
        "w2": ["a", 0.34]
    })
    print(polyCore)

    model = poly_to_cplex_model(polyCore)
    solution = model.solve()  ## Needs for usage IBM ILOG CPLEX Optimization Studio !

    # Print the solution
    if solution:
        print(f"Objective value: {solution.objective_value}")
        for color in polyCore.colors:
            print(f"x1 = {solution[color]}")
    else:
        print("No solution found")
