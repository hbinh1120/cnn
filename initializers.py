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

    def cell(self):
        '''
        creates a cell (a small model) with skip connections \n
        returns a list of nodes and their connection with eachother
        '''
        import random
        num_nodes = random.randint(self.config['min_nodes_in_cell'], self.config['max_nodes_in_cell'])
        nodes = []

        #each node takes 2 of previous nodes as inputs
        for i in range(num_nodes):
            edges = []
            for j in range(i + 2):
                for operation in self.config['operations']:
                    has_param = False
                    for param in self.config['config'][operation]:
                        has_param = True
                        for choice in self.config['config'][operation][param]:
                            edge = {}
                            edge['from_node'] = j
                            edge['type'] = operation
                            edge[param] = choice
                            edges.append(edge)
                            edges.append(edge)
                    if not has_param:
                        edge = {}
                        edge['from_node'] = j
                        edge['type'] = operation
                        edges.append(edge)
                        edges.append(edge)

            node = {}
            node['edges'] = edges
            node['graph'] = [0] * len(edges)
            for i in random.sample(range(len(edges)), 2):
                node['graph'][i] = 1
            node['combine_method'] = random.choice(self.config['combine_methods'])
            nodes.append(node)

        return nodes