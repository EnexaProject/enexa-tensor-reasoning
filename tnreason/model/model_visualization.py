import networkx as nx

import os
from matplotlib import pyplot as plt
from matplotlib import animation as animation

from tnreason.logic import expression_utils as eu


def visualize_model(expressionsDict,
                    factsDict={},
                    strengthMultiplier=4,
                    strengthCutoff=10,
                    fontsize=10,
                    showFormula=False,
                    evidenceDict={},
                    pos=None,
                    savePath=None,
                    show=True,
                    title="Visualization of the MLN"):
    expressionsList = [expressionsDict[key][0] for key in expressionsDict]
    constraintsList = [factsDict[key] for key in factsDict]

    atomsList = list(set(eu.get_all_variables(expressionsList)) | set( eu.get_all_variables(constraintsList)))

    ## Collect edges for position optimization
    edges = []
    for expressionKey in expressionsDict:
        for atom in eu.get_variables(expressionsDict[expressionKey][0]):
            edges.append([atom, expressionKey])

    constEdges = []
    for factsKey in factsDict:
        for atom in eu.get_variables(factsDict[factsKey]):
            constEdges.append([atom, factsKey])

    graph = nx.Graph()
    graph.add_nodes_from(atomsList)
    graph.add_nodes_from(expressionsDict.keys())
    graph.add_nodes_from(factsDict.keys())
    graph.add_edges_from(edges)
    graph.add_edges_from(constEdges)
    if pos is None:
        pos = nx.spring_layout(graph, k=0.8)

    ## Draw Nodes
    trueColor = "blue"
    falseColor = "red"

    # Known Trues
    atomFontSize = fontsize * 100
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in evidenceDict if bool(evidenceDict[atomKey])],
                           node_color=trueColor,
                           node_size=atomFontSize,
                           alpha=0.6)
    # Known False
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in evidenceDict if not bool(evidenceDict[atomKey])],
                           node_color=falseColor,
                           node_size=atomFontSize,
                           alpha=0.6)

    # Unknown Atoms
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=[atomKey for atomKey in atomsList if atomKey not in evidenceDict.keys()],
                           node_color="grey",
                           node_size=atomFontSize,
                           alpha=0.2)
    nx.draw_networkx_labels(graph, pos, {atomKey: atomKey for atomKey in atomsList}, font_size=fontsize)

    if showFormula:
        expressionLabels = {**{expressionKey: expressionsDict[expressionKey][0] for expressionKey in expressionsDict},
        **{factKey: factsDict[factKey] for factKey in factsDict}}
    else:
        expressionLabels = {**{expressionKey: expressionKey for expressionKey in expressionsDict},
        **{factKey: factKey for factKey in factsDict}}
    nx.draw_networkx_labels(graph, pos, expressionLabels, font_size=fontsize)

    ## Draw Edges
    colorList = ["red", "green", "blue", "purple", "orange", "yellow", "lime", "teal", "skyblue", "lightblue",
"maroon", "navyblue", "black", "white"]
    for i, expressionKey in enumerate(expressionsDict.keys()):
        strength = min(expressionsDict[expressionKey][1], strengthCutoff)
        drawEdges = []
        for atom in eu.get_variables(expressionsDict[expressionKey][0]):
            drawEdges.append([atom, expressionKey])
        nx.draw_networkx_edges(graph, pos,
                               edgelist=drawEdges,
                               width=strengthMultiplier * strength,
                               alpha=0.2,
                               edge_color=colorList[i])

    for i, factKey in enumerate(factsDict.keys()):
        strength = 0.8 * strengthMultiplier * strengthCutoff
        drawEdges = []
        for atom in eu.get_variables(factsDict[factKey]):
            drawEdges.append([atom, factKey])
        nx.draw_networkx_edges(graph, pos,
                               edgelist=drawEdges,
                               width = strength,
                               alpha=0.2,
                               edge_color="black")

    plt.title(title, fontsize=15)
    if savePath is not None:
        plt.savefig(savePath)
    if show:
        plt.show()
    return pos


def create_animation(pngDirPath, savePath):
    images = []
    for filename in os.listdir(pngDirPath):
        if filename.endswith('.png'):
            images.append(plt.imread(os.path.join(pngDirPath, filename)))
    # Create an animation
    fig = plt.figure()
    im = plt.imshow(images[0])
    ani = animation.FuncAnimation(fig, lambda i: im.set_array(images[i]), frames=len(images))

    # Save the animation as an MP4 file
    ani.save(savePath, writer='ffmpeg', fps=0.5)


if __name__ == "__main__":
    #create_animation("./demonstration/visualizations/",savePath="./animation.mp4")

    exDict = {
        "e0": ["a2", 1],
        "e1": [[["a2", "and", ["not", "a3"]], "and", ["a6", "and", ["not", "a7"]]], 5],
        "e2": [["a4", "and", ["not", "a2"]], 2],
        "e4": ["a5", 1],
        "e5": ["a5", 2]
    }

    facDict = {
        "c1": ["a10","and","a2"]
    }

    visualize_model(exDict, facDict, evidenceDict={"a2": 1, "a3": 0})

    exit()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.animation as animation

    img = []  # some array of images
    frames = []  # for storing the generated images
    fig = plt.figure()
    for i in range(6):
        frames.append([plt.imshow(img[i], cmap=cm.Greys_r, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    # ani.save('movie.mp4')
    plt.show()






    G = nx.Graph()
    G.add_edges_from([(12, 2)])  # , (2, 3), (3, 4), (4, 1)])

    pos = {12: [10, 0], 2: [1, 0]}
    nx.draw(G, pos=pos)
    plt.show()
