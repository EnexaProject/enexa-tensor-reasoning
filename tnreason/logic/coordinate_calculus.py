import numpy as np

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']


class CoordinateCore:
    def __init__(self, core_values, core_colors, name=None):
        self.values = core_values
        self.colors = core_colors
        self.name = name

        if len(core_colors) != len(core_values.shape):
            raise TypeError("Number of Colors does not match the Value Shape in Core {}!".format(name))

        ## We allow multiple colors for now, although there might be problems
        #if len(core_colors) != len(set(core_colors)):
        #    raise TypeError("There are duplicate colors in Core {}!".format(name))

    def clone(self):
        return CoordinateCore(np.copy(self.values), self.colors.copy(), self.name)

    def negate(self, ignore_ones=False):
        newName = ["not", self.name]

        if ignore_ones:
            return CoordinateCore(np.copy(- self.values), self.colors.copy(), newName)
        else:
            return CoordinateCore(np.ones(self.values.shape) - np.copy(self.values), self.colors.copy(), newName)

    def multiply(self, factor):
        return CoordinateCore(factor*self.values, self.colors, self.name)

    def compute_and(self, in_core1):
        core0 = self.clone()
        core1 = in_core1.clone()

        colorDict = {core0.colors[i]: alphabet[i] for i in range(len(core0.colors))}
        i = len(core0.colors)
        for color in core1.colors:
            if color not in colorDict:
                colorDict[color] = alphabet[i]
                i += 1

        core0_string = "".join([colorDict[color] for color in core0.colors])
        core1_string = "".join([colorDict[color] for color in core1.colors])
        premise_string = ",".join([core0_string, core1_string])

        ## Modified to handle multiply colors in a core
        head_string = ""
        out_colors = []
        for color in core0.colors:
            if colorDict[color] not in head_string:
                head_string += colorDict[color]
                out_colors.append(color)

        for color in core1.colors:
            if colorDict[color] not in head_string:
                head_string += colorDict[color]
                out_colors.append(color)
        contraction_string = "->".join([premise_string, head_string])

        return CoordinateCore(np.einsum(contraction_string, core0.values, core1.values),
                              out_colors,
                              [core0.name, "and", core1.name])

    def reorder_colors(self, newColors):
        oldColors = self.colors.copy()
        oldValues = np.copy(self.values)

        colorDict = {oldColors[i]: alphabet[i] for i in range(len(oldColors))}
        old_string = "".join([colorDict[color] for color in oldColors])
        new_string = "".join([colorDict[color] for color in newColors])
        contraction_string = "->".join([old_string, new_string])

        newValues = np.einsum(contraction_string, oldValues)

        self.values = newValues
        self.colors = newColors

    def sum_with(self, core1):
        if core1.colors != self.colors:
            self.reorder_colors(core1.colors)
        if core1.values.shape != self.values.shape:
            raise TypeError("Shapes do not match for summations!")
        return CoordinateCore(self.values + core1.values, self.colors, str(self.name) + "+" + str(core1.name))

    def compute_or(self, core1):
        core0 = self.extend_colors(core1)
        core1 = core1.extend_colors(self)

        summed = core0.sum_with(core1)
        truth_positions = np.argwhere(summed.values > 0)
        orValues = np.zeros(shape=summed.values.shape)
        for pos in truth_positions:
            orValues[tuple(pos)] = 1
        return CoordinateCore(orValues, summed.colors, [self.name, "and", core1.name])

    def extend_colors(self, core1):
        core0 = self.clone()

        colorDict = {core0.colors[i]: alphabet[i] for i in range(len(core0.colors))}
        i = len(core0.colors)
        for color in core1.colors:
            if color not in colorDict:
                colorDict[color] = alphabet[i]
                i += 1

        added_colors = []
        added_shape = []
        for i, color in enumerate(core1.colors):
            if not color in core0.colors:
                added_colors.append(color)
                added_shape.append(core1.values.shape[i])

        core0_string = "".join([colorDict[color] for color in core0.colors])
        added_string = "".join([colorDict[color] for color in added_colors])

        premise_string = ",".join([core0_string, added_string])
        head_string = core0_string + added_string
        contraction_string = "->".join([premise_string, head_string])
        newValues = np.einsum(contraction_string, core0.values, np.ones(shape=added_shape))

        return CoordinateCore(newValues, core0.colors + added_colors, name=str(core0.name) + "_extended")

    def contract_common_colors(self, in_core1, exceptions=[]):
        core0 = self.clone()
        core1 = in_core1.clone()

        colorDict = {core0.colors[i]: alphabet[i] for i in range(len(core0.colors))}
        i = len(core0.colors)
        for color in core1.colors:
            if color not in colorDict:
                colorDict[color] = alphabet[i]
                i += 1
            elif color in exceptions:
                cor_color = color + "_cor"
                colorDict[cor_color] = alphabet[i]
                i += 1

        core0_string = "".join([colorDict[color] for color in core0.colors])
        core1_string = ""
        new_string = ""
        new_colors = []
        for color in core1.colors:
            if color in exceptions:
                core1_string = core1_string + colorDict[color + "_cor"]
            else:
                core1_string = core1_string + colorDict[color]

            if color in exceptions:
                new_string = new_string + colorDict[color] + colorDict[color + "_cor"]
                new_colors.append(color)
                new_colors.append(color + "_cor")
            elif color not in core0.colors:
                new_string = new_string + colorDict[color]
                new_colors.append(color)

        for color in core0.colors:
            if color not in core1.colors:
                new_string = new_string + colorDict[color]
                new_colors.append(color)

        premise_string = ",".join([core0_string, core1_string])
        contraction_string = "->".join([premise_string, new_string])

        newValues = np.einsum(contraction_string, core0.values, core1.values)

        return CoordinateCore(newValues, new_colors, [core0.name, "contracted", core1.name])

    def create_constant(self, variableColors, zero=False):
        constantShape = []
        constantColors = []
        for i, color in enumerate(self.colors):
            if color not in variableColors:
                constantShape.append(self.values.shape[i])
                constantColors.append(color)
        if zero:
            return CoordinateCore(np.zeros(constantShape), constantColors, "Constant")
        else:
            return CoordinateCore(np.ones(constantShape), constantColors, "Constant")

    def reduce_color(self, conColor):
        ## To be implemented for efficiency increase: Alternative using np.sum
        if conColor not in self.colors:
            raise ValueError("Color {} not found in core {} for reduction.".format(conColor, self.name))
        colorDict = {col: alphabet[i] for i, col in enumerate(self.colors)}
        contractionstring = "".join([colorDict[col] for col in self.colors]) + "," + colorDict[
            conColor] + "->" + "".join(([colorDict[col] for col in self.colors if col != conColor]))
        onesValues = np.ones(self.values.shape[self.colors.index(conColor)])
        newValues = np.einsum(contractionstring, self.values, onesValues)
        return CoordinateCore(newValues, [col for col in self.colors if col != conColor])

    ## For usage in model (former BasisCalculus methods)
    def weighted_exponentiation(self, weight):
        return CoordinateCore(np.exp(weight * self.values), self.colors)

    def count_satisfaction(self):
        return np.sum(self.values) / np.prod(self.values.shape)

    def normalize(self, sum=1):
        return CoordinateCore(self.values * (sum / np.sum(self.values)), self.colors, self.name)

    def exponentiate(self):
        return CoordinateCore(np.exp(self.values), self.colors, self.name)

    def logarithm(self):
        return CoordinateCore(np.log(self.values), self.colors, self.name)


## Small Test Skript
if __name__ == "__main__":
    core0_values = np.random.normal(size=(100, 10, 5))
    core0_colors = ["x1", "y2", "z3"]

    core0 = CoordinateCore(core0_values, core0_colors, "Sledz")
    # core0.reorder_colors(["b", "c", "a"])

    core1_values = np.random.normal(size=(10, 100, 5))
    core1_colors = ["y2", "x1", "z3"]

    core0.negate()
    core1 = CoordinateCore(core1_values, core1_colors, "Jaszczur")

    einSum_Core = core0.compute_and(core1)
    compare_Core = core0.compute_and(core1).negate(ignore_ones=True)

    toBeZero = einSum_Core.sum_with(compare_Core)

    print(np.linalg.norm(einSum_Core.values))
    print(np.linalg.norm(compare_Core.values))
    print(np.linalg.norm(toBeZero.values))
    print(einSum_Core.colors)
    print(compare_Core.colors)
    core0.sum_with(core1)
