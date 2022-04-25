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
            operations = []
            for _ in range(2):
                op = {}
                type = random.choice(self.config['operations'])
                op['type'] = type
                for key in self.config['config'][type]:
                    op[key] = random.choice(self.config['config'][type][key])
                operations.append(op)

            node = {}
            node['inputs'] = random.choices(list(range(-2, i)), k=2)
            node['operations'] = operations
            node['combine_method'] = random.choice(self.config['combine_methods'])
            nodes.append(node)

        return nodes