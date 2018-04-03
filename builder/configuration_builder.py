from data.configuration import Configuration


def build_config_from_json(file) -> Configuration:
    config = Configuration()
    config.input_layer_shape = (file['input_layer']['height'], file['input_layer']['width'])
    config.output_layer_classes = file['output_layer']['classes']
    config.input_layer_type = file['input_layer']['type']
    config.output_layer_type = file['output_layer']['type']
    config.hidden_layers_nodes = file['hidden_layers_nodes']
    config.epoch = file['epoch']
    config.learning_rate = file['learning_rate']
    config.batch_size = file['batch_size']
    config.num_hidden_layer_nodes = len(config.hidden_layers_nodes)

    return config