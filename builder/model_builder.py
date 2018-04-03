from data.configuration import Configuration
from data.model import Model


def build_model_from_configuration(config: Configuration):
    model = Model()
    model.inp_nodes = config.input_layer_shape