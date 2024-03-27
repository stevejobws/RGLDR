import networkx as nx
import random

def metapath_random_walk(G, start_node, metapath, walk_length):
    """
    Perform a random walk based on the given metapath.

    Args:
    G (networkx.Graph): Input graph
    start_node (int or str): Start node for the random walk
    metapath (str): Metapath schema (e.g., "DPTDP")
    walk_length (int): Length of the random walk

    Returns:
    list: A sequence of nodes visited during the random walk
    """
    walk = [start_node]
    cur_node = start_node
    cur_metapath_idx = 0

    while len(walk) < walk_length:
        # Get neighbors with the desired node type
        cur_type = metapath[cur_metapath_idx % len(metapath)]
        next_type = metapath[(cur_metapath_idx + 1) % len(metapath)]
        next_candidates = [n for n in G.neighbors(cur_node) if G.nodes[n]['node_type'] == next_type]

        if not next_candidates:
            break

        # Choose a random neighbor
        next_node = random.choice(next_candidates)
        walk.append(next_node)
        cur_node = next_node
        cur_metapath_idx += 1

    return walk

def generate_subgraph(G, start_node, metapath, walk_length, num_walks):
    """
    Generate a subgraph by performing multiple random walks based on the metapath.

    Args:
    G (networkx.Graph): Input graph
    start_node (int or str): Start node for the random walks
    metapath (str): Metapath schema (e.g., "DPTDP")
    walk_length (int): Length of the random walks
    num_walks (int): Number of random walks to perform

    Returns:
    networkx.Graph: A subgraph generated from the random walks
    """
    subgraph_nodes = set()
    for _ in range(num_walks):
        walk = metapath_random_walk(G, start_node, metapath, walk_length)
        subgraph_nodes.update(walk)

    return G.subgraph(subgraph_nodes)
    
def generate_integer_list(start, end):

    return list(range(start, end))

def subgraph_adj_matrix_with_original_shape(G, subgraph):

    n = len(G.nodes())
    subgraph_nodes = subgraph.nodes()
    full_adj_matrix = nx.adjacency_matrix(G).todense()
    subgraph_adj_matrix = np.zeros((n, n))

    for i in subgraph_nodes:
        for j in subgraph_nodes:
            subgraph_adj_matrix[i, j] = full_adj_matrix[i, j]

    return subgraph_adj_matrix
    
import networkx as nx
from tqdm import tqdm

dataset = 'B-Dataset'# B-dataset C-dataset F-dataset
drdi = pd.read_csv('./data/'+dataset+'/DrDiNum.csv', header = None)#, sep='\t', header = None)
drpr = pd.read_csv('./data/'+dataset+'/DrPrNum.csv', header = None)#, sep='\t', header = None)
dipr = pd.read_csv('./data/'+dataset+'/DiPrNum.csv', header = None)#, sep='\t', header = None)
allnode= pd.concat([drdi,drpr,dipr])
max_node = max([max(allnode[0]),max(allnode[1])])

G = nx.Graph()
G.add_nodes_from(generate_integer_list(0,max(drdi[0])+1), node_type="D")
G.add_nodes_from(generate_integer_list(max(drdi[0])+1,max(drdi[1])+1), node_type="P")
G.add_nodes_from(generate_integer_list(max(drdi[1])+1,max_node+1), node_type="T")
drdipr = pd.concat([drdi,drpr,dipr])
edges = [(row[0], row[1]) for _, row in drdipr.iterrows()]
G.add_edges_from(edges)


metapaths = ["DTP", "DTDP", "DTPTD"]
walk_length = 10 # 20
num_walks = 20 # 5
# max_node = G.number_of_nodes()
i = 0
for metapath in metapaths:
    i +=1
    subgraphs = {}
    for start_node in tqdm(G.nodes(), desc="Generating subgraphs"):
        subgraph = generate_subgraph(G, start_node, metapath, walk_length, num_walks)
        subgraphs[start_node] = subgraph

    combined_subgraph = nx.Graph()
    for _, subgraph in subgraphs.items():
        combined_subgraph = nx.compose(combined_subgraph, subgraph)

    combined_subgraph_adj_matrix = subgraph_adj_matrix_with_original_shape(G, combined_subgraph)

    combined_subgraph_adj_matrix_df = pd.DataFrame(combined_subgraph_adj_matrix)
    combined_subgraph_adj_matrix_df.to_csv("./data/"+dataset+"/Generating_subgraphs_adj_matrix_"+str(i)+".csv", index=False, header=False)