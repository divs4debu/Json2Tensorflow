import tensorflow as tf

class Model:

    def __init__(self):
        self.inp_nodes = 0
        self.out_nodes = 0
        self.hidden_layers_nodes = []
        self.num_hidden_layers = 0
        self.input_placeholder = tf.placeholder()
        self.output_placeholder = tf.placeholder()

    def __str__(self) -> str:
        return 'input_nodes: ' + str(self.inp_nodes) + \
               '\noutput_nodes: ' + str(self.out_nodes) + \
               '\nhidden_layers' + str(self.hidden_layers_nodes) + \
               '\ninput_placeholder: ' + self.input_placeholder + \
               '\noutput_placeholer: ' + self.output_placeholder
