import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """
    load data set
    :return: label, x
    """
    file_path = './data/tiny_train_input.csv'
    data = pd.read_csv(file_path, header=None)
    m, n = data.shape
    label = data[0].values
    label = np.array(label).reshape([m, 1])
    x = data.iloc[:, 1:n].values
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return label, x


def linear_weight(n):
    """
    init linear term weight and bias
    :param n:
    :return: w, b
    """
    w = tf.Variable(tf.truncated_normal(shape=[n, 1], dtype=tf.float32, mean=0.0, stddev=0.01), name="linear_weight")
    b = tf.Variable(tf.zeros(1, dtype=tf.float32), name="bias")
    return w, b


def lr(x_train, linear_w, b):
    """
    output
    :param x_train:
    :param linear_w:
    :param b:
    :return: y_hat
    """
    output = tf.add(b, tf.matmul(x_train, linear_w))
    y_hat = tf.nn.sigmoid(output)
    return y_hat


def loss(y_hat, y_train, linear_w, lambda_w=0.1):
    """
    loss function
    :param y_hat:
    :param y_train:
    :param linear_w:
    :param lambda_w:
    :return: train set loss and log loss
    """
    log_loss = -tf.reduce_mean(
        y_train * tf.log(y_hat + 1e-24) + (1 - y_train) * tf.log(1 - y_hat + 1e-24)
    )
    l2_norm = lambda_w * tf.reduce_sum(
        tf.pow(linear_w, 2)
    )
    train_loss = tf.add(
        log_loss,
        l2_norm
    )
    return train_loss, log_loss


def train(label, x, epochs=1000, learning_rate=0.000002):
    """
    training lr  model
    :param label:
    :param x:
    :param epochs:
    :param learning_rate:
    :return:
    """
    m, n = x.shape
    x_train = tf.placeholder(shape=[None, n], dtype=tf.float32)
    y_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    linear_w, b = linear_weight(n)
    y_hat = lr(x_train, linear_w, b)
    train_loss, log_loss = loss(y_hat, y_train, linear_w)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(train_loss)
    init = tf.global_variables_initializer()
    train_loss_list = np.zeros(epochs)
    log_loss_list = np.zeros(epochs)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            _, train_loss_, train_log_loss, y_output,linear_w_1 = sess.run([train_op, train_loss, log_loss, y_hat, linear_w],
                                                                feed_dict={x_train: x, y_train: label})
            train_loss_list[epoch] = train_loss_
            log_loss_list[epoch] = train_log_loss
    return train_loss_list, log_loss_list, y_output,linear_w_1


def main():
    """
    main function
    :return: rain_loss_list, log_loss_list
    """
    label, x = load_data()
    train_loss_list, log_loss_list, y_output,linear_w_1 = train(label, x)
    return train_loss_list, log_loss_list, y_output,linear_w_1


if __name__ == "__main__":
    train_loss_list, log_loss_list, y_output,linear_w_1 = main()
    plt.plot(range(1000), train_loss_list)
    plt.show()
