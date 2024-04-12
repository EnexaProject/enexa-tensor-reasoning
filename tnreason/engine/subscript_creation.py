def get_colorDict(nestedColorsList, symbols):
    colorDict = {}
    i = 0
    for colors in nestedColorsList:
        for color in colors:
            if color not in colorDict:
                if i >= len(symbols):
                    raise ValueError("Length of Contraction is too large for Einsum!")
                colorDict[color] = symbols[i]
                i += 1
    return colorDict

def get_substring(conCoreDict, openVariables, symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):

    colorDict = get_colorDict([conCoreDict[key].colors for key in conCoreDict], symbols=symbols)
    coreOrder = list(conCoreDict.keys())
    colorOrder = list(colorDict.keys())
    leftString = ",".join([
        "".join([colorDict[color] for color in conCoreDict[key].colors])
        for key in coreOrder
    ])
    rightString = "".join([colorDict[color] for color in colorOrder if color in openVariables])

    return "->".join([leftString, rightString]), coreOrder, colorDict, colorOrder