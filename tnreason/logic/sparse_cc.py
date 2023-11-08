import scipy.sparse as scs

class SparseCoordinateCore:

    def __init__(self, core_values, core_colors, name=None):
        self.values = core_values
        self.colors = core_colors
        self.name = name

        if len(core_colors) != len(core_values.shape):
            raise TypeError("Number of Colors provided does not match the Value Shape!")

    ## To be implemented:
    # and, not, sum, solve operations on sparse coordinates