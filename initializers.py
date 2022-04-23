'''
particle initializers \n
each function returns a (cnn_layers, fc_layers) tuple \n
this tuple does not include an output layer and a global pooling / flatten layer
'''
def sequential(config_file='config.json'):
    '''
    add layers to the list of layers, following transition rules as defined in config_file
    '''
    import json
    import os.path
    import random

    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

    num_cnn_layers = random.randint(config['min_cnn_layers'], config['max_cnn_layers'])
    num_fc_layers = random.randint(config['min_fc_layers'], config['max_fc_layers'])
    cnn_layers = []
    fc_layers = []

    next_layer_type = 'conv'
    for _ in range(num_cnn_layers):
        layer = {}
        layer['type'] = next_layer_type
        for key in config['config'][next_layer_type]:
            layer[key] = random.choice(config['config'][next_layer_type][key])
        cnn_layers.append(layer)
        next_layer_type = random.choices(config['layers'], config['transitions'][next_layer_type])[0]

    next_layer_type = 'fc'
    for _ in range(num_fc_layers):
        layer = {}
        layer['type'] = next_layer_type
        for key in config['config'][next_layer_type]:
            layer[key] = random.choice(config['config'][next_layer_type][key])
        fc_layers.append(layer)
        next_layer_type = random.choices(config['layers'], config['transitions'][next_layer_type])[0]

    return cnn_layers, fc_layers