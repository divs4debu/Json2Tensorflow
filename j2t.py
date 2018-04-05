# import tensorflow
import tensorflow as tf
import json


def read_json_file(file_path):
    with open(file_path) as f:
        config = json.loads(f.read())
    return config


config = read_json_file("xor.json")

# defining data variables

x_data = config["input_data"]
y_data = config["output_data"]

# defining placeholders for data input and output

x_input = tf.placeholder(tf.float32)
y_input = tf.placeholder(tf.float32)

# defining hyper-parameters

epochs = config['epochs']
learning_rate = config['learning_rate']

n_inputs = config['input_features']
n_hidden = config['hidden_layers']
n_output = config['output']

layout = [n_inputs] + n_hidden + [n_output]

print("x_data: ", x_data, "\ny_data: ", y_data, "\nepochs: ", epochs, "\nlearning_rate: ",
      learning_rate, "\nn_inputs: ", n_inputs, "\nn_outputs: ", n_output, "\nlayout: ", layout)

# creating model

_layered_model = []

for i in range(len(layout) - 1):
    _layered_model.append({'weights': tf.Variable(tf.random_uniform([layout[i], layout[i + 1]], -1.0, 1.0)),
                           'biases': tf.Variable(tf.zeros([layout[i + 1]]), name="Bias" + str(i + 1))})
print("Layer description: ",_layered_model)

# creating computation graph

_computation_graph = []
for i in range(len(_layered_model)):
    if i is 0:
        if n_inputs == 1:
            _computation_graph.append(
                tf.sigmoid(tf.add((x_input *_layered_model[i]['weights']), _layered_model[i]['biases'])))
        else:
            _computation_graph.append(tf.sigmoid(tf.add(tf.matmul(x_input,  _layered_model[i]['weights']), _layered_model[i]['biases'])))
    else:
        _computation_graph.append(
            tf.sigmoid(tf.add(tf.matmul(_computation_graph[i - 1], _layered_model[i]['weights']), _layered_model[i]['biases'])))

output = _computation_graph[len(_computation_graph)-1]

print("Output_node: ", output)

# train

cost =  tf.reduce_mean(-y_input*tf.log(output) - (1 - y_input)*tf.log(1 - output), name="cost")
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        sess.run([optimizer, cost], feed_dict={x_input: x_data, y_input: y_data})

        if i % 50 == 0:
            print(sess.run(cost, feed_dict={x_input: x_data, y_input: y_data}))

    value = tf.round(sess.run(output, feed_dict={x_input: x_data}))
    print(sess.run(value))