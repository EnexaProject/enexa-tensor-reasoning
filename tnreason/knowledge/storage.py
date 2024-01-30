import yaml

def save_as_yaml(modelSpec, savePath):
    with open(savePath, "w") as file:
        yaml.dump(modelSpec, file)

def load_from_yaml(loadPath):
    with open(loadPath) as file:
        data = yaml.safe_load(file)
    return data


