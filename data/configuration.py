class Configuration:
    def __init__(self):
        self.input_layer_shape = tuple()
        self.input_layer_type = ""
        self.hidden_layers_nodes = []
        self.num_hidden_layer_nodes = 0
        self.output_layer_classes = 0
        self.output_layer_type = ""
        self.batch_size = 1
        self.learning_rate = 0.01
        self.epoch = 1
