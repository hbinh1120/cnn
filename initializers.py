'''
particle initializers \n
'''
class Initializers:
    def __init__(self, config_file='config.json'):
        import json
        import os.path
        
        self.config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)

    def sequential(self):
        '''
        creates a sequential model following transition rules \n
        returns tuple (cnn_layers, fc_layers) which are lists of layers
        '''
        import random
        num_cnn_layers = random.randint(self.config['min_cnn_layers'], self.config['max_cnn_layers'])
        num_fc_layers = random.randint(self.config['min_fc_layers'], self.config['max_fc_layers'])
        cnn_layers = []
        fc_layers = []

        next_layer_type = 'conv'
        for _ in range(num_cnn_layers):
            layer = {}
            layer['type'] = next_layer_type
            for key in self.config['config'][next_layer_type]:
                layer[key] = random.choice(self.config['config'][next_layer_type][key])
            cnn_layers.append(layer)
            next_layer_type = random.choices(self.config['layers'], self.config['transitions'][next_layer_type])[0]

        next_layer_type = 'fc'
        for _ in range(num_fc_layers):
            layer = {}
            layer['type'] = next_layer_type
            for key in self.config['config'][next_layer_type]:
                layer[key] = random.choice(self.config['config'][next_layer_type][key])
            fc_layers.append(layer)
            next_layer_type = random.choices(self.config['layers'], self.config['transitions'][next_layer_type])[0]

        return cnn_layers, fc_layers

    def cell_graph(self):
        '''
        returns a list of all possible connections in a cell \n
        each element is a list of all possible inputs for the corresponding node
        '''
        import copy
        edges = []
        for i in range(self.config['num_nodes']):
            node_edges = []
            for j in range(i + 2):
                for operation in self.config['operations']:
                    edge = copy.deepcopy(operation)
                    edge['from_node'] = j
                    node_edges.append(edge)
            edges.append(node_edges)
        return edges

    def cell_init(self, edges):
        '''
        returns a subgraph of the full graph of all possible connections \n
        subgraph contains exactly 2 in edges for each node
        '''
        import random
        graph = []
        for node_edges in edges:
            graph_node = []
            graph_node = [0] * len(node_edges)
            for i in random.sample(range(len(node_edges)), 2):
                graph_node[i] = 1
            graph.append(graph_node)
        return graph