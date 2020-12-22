import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl


def draw_cognitive_map(weights, title=None, save_path=None):
    colormap = plt.cm.bwr_r
    plt.figure(figsize=(14, 10))
    if title:
        plt.title(title)
    graph = nx.DiGraph()
    if len(weights) != len(weights[0]):
        print(f"Unable to draw graph: weights is {len(weights)}x{len(weights[0])}, not square");
        return
    edges = []
    max_weight_abs = 0.0
    edge_colors = []
    edge_alphas = []
    labels = {}
    for i in range(len(weights)):
        for j in range(len(weights)):
            weight_abs = abs(weights[i][j])
            if weight_abs > 0.2:
                edges.append((i, j, {'weight': f"{weights[i][j]}"}))
                edge_colors.append(weights[i][j])
                edge_alphas.append(weights[i][j])
            if weight_abs > max_weight_abs:
                max_weight_abs = weight_abs
        labels[i] = f"{i}\n{weights[i][i]:.2f}"
    graph.add_edges_from(edges)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos,node_size=600, node_color="black")
    nx.draw_networkx_labels(graph,pos,labels=labels,font_size=10,font_family='sans-serif', font_color='white')
    edges = nx.draw_networkx_edges(
        graph,
        pos,
        node_size=600,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color=edge_colors,
        edge_cmap=colormap,
        width=2,
        connectionstyle='arc3, rad = 0.2'
    )
    # for i in range(len(edges)):
    #     edges[i].set_alpha(edge_alphas[i])
    pc = mpl.collections.PatchCollection(edges, cmap=colormap)
    pc.set_array(edge_colors + [max_weight_abs, -max_weight_abs])
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()