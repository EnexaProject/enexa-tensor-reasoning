import numpy as np


class BasisCore:
    def __init__(self, core_values, core_colors, headcolor=None, name=None):
        self.values = core_values

        self.colors = core_colors

        if headcolor is None:
            self.headcolor = self.colors[0]
        else:
            self.headcolor = headcolor
        self.name = name

    def clone(self):
        return BasisCore(np.copy(self.values), self.colors.copy(), self.headcolor, self.name)

    def negate(self):
        core0 = self.clone()
        newName = ["not", core0.name]

        newValues = np.tensordot(np.copy(core0.values), create_negation_tensor(),
                                 axes=([core0.colors.index(core0.headcolor)], [0]))

        newColors = core0.colors.copy()
        newColors.pop(newColors.index(core0.headcolor))
        newColors.append(core0.headcolor)

        newHeadColor = core0.headcolor

        return BasisCore(newValues, newColors, newHeadColor, newName)

    def compute_and(self, core1):
        core0 = self.clone()
        newName = [core0.name, "and", core1.name]

        if core0.headcolor == "TruthEvaluated" or core1.headcolor == "TruthEvaluated":
            raise TypeError("Basis Core does not have a Headcolor since already evaluated.")

        contracted = np.tensordot(core0.values, create_and_tensor(), axes=([core0.colors.index(core0.headcolor)], [0]))
        contracted = np.tensordot(contracted, core1.values, axes=([-2], [core1.colors.index(core1.headcolor)]))

        contracted_colors = core0.colors
        contracted_colors.pop(contracted_colors.index(core0.headcolor))
        contracted_colors.append(core0.headcolor)

        core1_colors = core1.colors.copy()
        core1_colors.pop(core1.colors.index(core1.headcolor))

        contracted_colors = contracted_colors + core1_colors

        return BasisCore(contracted, contracted_colors, core0.headcolor, newName)

    def calculate_truth(self):
        truthvec = create_truth_vec()

        core0 = self.clone()
        contracted = np.tensordot(core0.values, truthvec, axes=([core0.colors.index(core0.headcolor)], [0]))
        contracted_colors = (core0.colors.copy())
        contracted_colors.pop(contracted_colors.index(core0.headcolor))

        return BasisCore(contracted, contracted_colors, "TruthEvaluated", core0.name)


def create_negation_tensor():
    negation_tensor = np.zeros((2, 2))
    negation_tensor[0, 1] = 1
    negation_tensor[1, 0] = 1
    return negation_tensor


def create_and_tensor():
    and_tensor = np.zeros((2, 2, 2))
    and_tensor[0, 0, 0] = 1
    and_tensor[0, 1, 0] = 1
    and_tensor[1, 0, 0] = 1
    and_tensor[1, 1, 1] = 1
    return and_tensor


# Not required!
def create_delta_tensor(order=3, legim=2):
    delta_tensor = np.zeros((tuple([2 for k in range(order)])))
    for i in range(legim):
        delta_tensor[tuple([i for k in range(order)])] = 1
    return delta_tensor

def create_truth_vec():
    truthvec = np.zeros(2)
    truthvec[1] = 1
    return truthvec