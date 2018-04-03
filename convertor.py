import json
import tensorflow as tf


def read_json_file(file_path):
    with open(file_path) as f:
        config = json.loads(f.read())
    return config


def create_model_from_configuration(c):
    hidden_layers = c['hidden_layers_nodes']
    output_layer = c['output_layer']['classes']
    features = c['input_layer']['width']
    batch_size = c['batch_size']

        x = tf.placeholder(c["input_layer"]["type"], [c["input_layer"]["height"], c["input_layer"]["width"]])
        y = tf.placeholder(c["output_layer"]["type"])

    nodes = [features] + hidden_layers + [output_layer]
    layers = []
    fine_layers = []
    for i in range(len(nodes) - 1):
        layers.append(
            {
                'weights': tf.Variable(tf.random_normal([nodes[i], nodes[i + 1]])),
                'biases': tf.Variable(tf.random_normal([nodes[i + 1]]))
            }
        )

    size = len(layers)
    for i in range(size):
        if i is 0:
            fine_layers.append(tf.add(tf.matmul(x, layers[0]['weights']), layers[0]['biases']))
        else:
            fine_layers.append(tf.add(tf.matmul(fine_layers[i - 1], layers[i]['weights']), layers[i]['biases']))
        if i is not size - 1:
            fine_layers[i] = tf.nn.relu(fine_layers[i])

    print(fine_layers)
    return fine_layers[len(fine_layers) - 1]


def train_neural_network(config):
    prediction = create_model_from_configuration(config)
    #  print(prediction)

    n_epochs = config['epoch']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    data = config['data']

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in (data['inputs']:
                _, c = sess.run([optimizer, cost], feed_dict={x: i, y: data['outputs'][i]})
            epoch_loss += c
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))





configuration = read_json_file("sample_input.json")
train_neural_network(configuration)
