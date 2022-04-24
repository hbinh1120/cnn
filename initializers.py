'''
particle initializers \n
each function returns a (cnn_layers, fc_layers) tuple \n
this tuple does not include an output layer and a global pooling / flatten layer
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
        creates a sequential model following transition rules
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