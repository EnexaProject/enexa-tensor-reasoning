def save_as_yaml(modelSpec, savePath):
    import yaml
    with open(savePath, "w") as file:
        yaml.dump(modelSpec, file)

def load_from_yaml(loadPath):
    import yaml
    with open(loadPath) as file:
        data = yaml.safe_load(file)
    return data