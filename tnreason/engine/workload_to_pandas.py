import pandas as pd

import numpy as np


def pandas_rencoding_from_function(inshape, outshape, incolors, outcolors, function, name="PolyEncoding"):
    df = pd.DataFrame(data=[i + tuple([int(entry) for entry in function(*i)] + [1]) for i in np.ndindex(*inshape)],
                      columns=incolors + outcolors + ["values"])
    return PandasCore(values=df, colors=incolors + outcolors, shape=inshape + outshape, valueColumn="values")


def pandas_tencoding_from_function(inshape, incolors, function, name="PolyEncoding"):
    df = pd.DataFrame(data=[i + (function(*i),) for i in np.ndindex(*inshape) if function(*i) != 0],
                      columns=incolors + ["values"])
    return PandasCore(values=df, colors=incolors, shape=inshape, valueColumn="values")


class PandasCore:

    def __init__(self, values, colors, name=None, shape=None, valueColumn="values", nanValue=-1):
        self.colors = colors
        self.name = name

        if shape is not None:
            self.shape = shape

        self.values = values
        self.valueColumn = valueColumn

        if not valueColumn in self.values.columns:
            self.values[valueColumn] = 1

        self.nanValue = nanValue
        self.values = self.values.fillna(nanValue)

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        checkDict = {color: item[i] for i, color in enumerate(self.colors)}
        value = 0
        for j, row in self.values.iterrows():
            if all([row[col] == checkDict[col] or row[col] == self.nanValue for col in self.colors]):
                value = value + row[self.valueColumn]
        return value

    def contract_with(self, core2):
        core2.values = core2.values.rename(columns={core2.valueColumn: self.valueColumn + "_sec"})
        colorsShapeDict = {**{color: self.shape[i] for i, color in enumerate(self.colors)},
                           **{color: core2.shape[i] for i, color in enumerate(core2.colors)}}
        preValues = self.values.merge(core2.values, how="cross") # Build the naive product
        for newColor in colorsShapeDict.keys(): # Drop zero rows
            if newColor in self.colors and newColor in core2.colors:
                preValues = preValues[preValues[newColor + "_x"] == preValues[newColor + "_y"]].drop(newColor + "_y",
                                                                                                     axis=1)
        contractedValues = preValues.rename(
            columns={col: col[:-2] for col in preValues.columns if col.endswith("_x") or col.endswith("_y")})
        contractedValues[self.valueColumn] = contractedValues[self.valueColumn] * contractedValues[
            self.valueColumn + "_sec"]
        contractedValues = contractedValues.drop(self.valueColumn + "_sec", axis=1)
        return PandasCore(values=contractedValues,
                          colors=list(colorsShapeDict.keys()),
                          shape=list(colorsShapeDict.values()),
                          valueColumn=self.valueColumn)

    def reduce_colors(self, newColors):
        ## Add correcting factors for trivial colors to be dropped
        self.values = self.values.reset_index()  # Unclear, in which cases this is necessary before the usage of loc
        for j in range(len(self.values)):
            self.values.loc[j, self.valueColumn] = np.prod([self.shape[k] for k, col in enumerate(self.colors) if
                                                            self.values.loc[
                                                                j, col] == self.nanValue and col not in newColors]) * \
                                                   self.values.loc[j, self.valueColumn]
        self.values = self.values.groupby(newColors)[self.valueColumn].sum().reset_index()
        self.colors = newColors

    def add_identical_slices(self):
        self.values = self.values.groupby(self.colors)[self.valueColumn].sum().reset_index()

    def multiply(self, weight):
        self.values[self.valueColumn] = weight * self.values[self.valueColumn]

    def sum_with(self, sumCore):
        sumCore.values = sumCore.values.rename(columns={sumCore.valueColumn: self.valueColumn})

        colorsShapeDict = {**{color: self.shape[i] for i, color in enumerate(self.colors)},
                           **{color: sumCore.shape[i] for i, color in enumerate(sumCore.colors)}}
        return PandasCore(values=pd.concat([self.values, sumCore.values], ignore_index=True),
                          colors=list(colorsShapeDict.keys()),
                          shape=list(colorsShapeDict.values()),
                          valueColumn=self.valueColumn)

    def enumerate_slices(self, enumerationColor="j"):
        self.values[enumerationColor] = [i for i in range(len(self.values))]
        self.colors = self.colors + [enumerationColor]
        self.shape = self.shape + [len(self.values)]

    def reorder_colors(self, newColors):
        if set(self.colors) == set(newColors):
            self.colors = newColors
        else:
            raise ValueError("Reordering of Colors in Core {} not possible, since different!".format(self.name))
