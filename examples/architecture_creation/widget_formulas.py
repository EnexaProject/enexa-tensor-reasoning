import ipywidgets as widgets
from IPython.display import display

from tnreason.encoding import neurons_to_cores as ntc


def find_selection_variables_neuron(neuronSpecList, neuronName):
    return {
        neuronName + ntc.connectiveSelColorSuffix: neuronSpecList[0],
        **{neuronName + "_" + ntc.posPrefix + str(i - 1) + ntc.candidatesColorSuffix: neuronSpecList[i] for i in
           range(1, len(neuronSpecList))}
    }


def find_selection_variables(architecture):
    selVarDict = {}
    for neuronName in architecture:
        selVarDict = {**selVarDict, **find_selection_variables_neuron(architecture[neuronName], neuronName)}
    return selVarDict


def generate_architecture_widget(architecture):
    selVarDict = find_selection_variables(architecture)
    widgetDict = {}
    for selVarKey in selVarDict:
        widgetDict[selVarKey] = widgets.Dropdown(options=selVarDict[selVarKey], description=selVarKey)
    output = widgets.Output()
    display(*[widgetDict[key] for key in widgetDict], widgets.Output())
    return widgetDict


def widgetDict_to_solution_expression(widgetDict, neuronDict, selVarDict):
    selectionDict = {selVarKey: selVarDict[selVarKey].index(widgetDict[selVarKey].value) for selVarKey in selVarDict}
    return ntc.create_solution_expression(neuronDict, selectionDict)


if __name__ == "__main__":
    sampleArchitecture = {"neur1": [
        ["imp", "eq"],
        ["a1", "a2"],
        ["a3", "a2"]
    ],
        "neur2": [
            ["not", "id"],
            ["neur1", "a2"]]
    }
    widgetDict = generate_architecture_widget(sampleArchitecture)
    solutionExpression = widgetDict_to_solution_expression(widgetDict, sampleArchitecture,
                                                           find_selection_variables(sampleArchitecture))
    print(solutionExpression)
