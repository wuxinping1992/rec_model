import tensorflow as tf
import pandas as pd
import numpy as np
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


def init_weight(n, p):
    """
    init weight
    :param n: feature dim
    :param p: latent vector dim
    :return:
    """
    linear_w = tf.Variable(tf.truncated_normal(shape=[n, 1], dtype=tf.float32, mean=0, stddev=0.01), name="linear_weight")
    V = tf.Variable(tf.truncated_normal(shape=[n, p], dtype=tf.float32, mean=0, stddev=0.01), name="cross_weight")
    b = tf.Variable(tf.zeros(1, dtype=tf.float32), name="bias")
    w_last = tf.Variable(tf.truncated_normal([1, 1], dtype=tf.float32, mean=0, stddev=0.01), name="last_layer")
    b_last = tf.Variable(tf.constant(0.01, dtype=tf.float32), name="b_last")
    return linear_w, V, b, w_last, b_last


def fm(x_train, linear_w, V, b, w_last, b_last):
    """
    FM model
    :param x_train:
    :param linear_w:
    :param V:
    :param b:
    :return:
    """
    linear_term = tf.add(
        b,
        tf.matmul(
            x_train,
            linear_w
        )
    )
    cross_term = tf.reduce_sum(
        tf.subtract(
            tf.pow(tf.matmul(x_train, V), 2),
            tf.matmul(tf.pow(x_train, 2), tf.pow(V, 2))
        ),
        axis=1, keepdims=True
    )
    output = tf.add(linear_term, cross_term)
    out = tf.add(b_last, tf.matmul(output, w_last))
    y_hat = tf.sigmoid(out)
    return y_hat


def loss(y_hat, y_train, linear_w, V, lambda_1=0.01, lambda_2=0.02):
    """
    loss function
    :param y_hat:
    :param y_train:
    :param linear_w:
    :param V:
    :param lambda_1:
    :param lambda_2:
    :return:
    """
    log_loss = -tf.reduce_mean(
        y_train * tf.log(y_hat + 1e-24) + (1 - y_train) * tf.log(1 - y_hat + 1e-24)
    )
    l2_norm = tf.add(
        lambda_1 * tf.reduce_sum(tf.pow(linear_w, 2)),
        lambda_2 * tf.reduce_sum(tf.pow(V, 2))
    )
    train_loss = tf.add(log_loss, l2_norm)
    return train_loss, log_loss


def train(label, x, p=5, epochs=1000, learning_rate=0.001):
    """
    training FFM model
    :param label:
    :param x:
    :param p:
    :param epochs:
    :param learning_rate:
    :return:
    """
    m, n = x.shape
    x_train = tf.placeholder(shape=[None, n], dtype=tf.float32)
    y_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    linear_w, V, b ,w_last, b_last = init_weight(n, p)
    y_hat = fm(x_train, linear_w, V, b, w_last, b_last)
    train_loss, log_loss = loss(y_hat, y_train, linear_w, V)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(train_loss)
    init = tf.global_variables_initializer()
    train_loss_list = np.zeros(epochs)
    log_loss_list = np.zeros(epochs)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            _, train_loss_, train_log_loss, y_output = sess.run([train_op, train_loss, log_loss, y_hat],
                                                                feed_dict={x_train: x, y_train: label})
            train_loss_list[epoch] = train_loss_
            log_loss_list[epoch] = train_log_loss
    return train_loss_list, log_loss_list, y_output


def main():
    """
    main function
    :return: train_loss_list, log_loss_list
    """
    label, x = load_data()
    train_loss_list, log_loss_list, y_output = train(label, x)
    return train_loss_list, log_loss_list, y_output


if __name__ == "__main__":
    train_loss_list, log_loss_list, y_output = main()
    print(y_output[:5])
    plt.plot(range(1000), train_loss_list)
    plt.show()
