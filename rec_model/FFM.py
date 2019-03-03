import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    file_path = './data/tiny_train_input.csv'
    data = pd.read_csv(file_path, header=None)
    m, n = data.shape
    label = data[0].values
    label = np.array(label).reshape([m, 1])
    x = data.iloc[:, 1:n].values
    x_filed = [i // 11 for i in range(n)]
    return label, x, x_filed


def init_weight(n, p, k):
    """
    init weight
    :param n: feature dim
    :param p: latent vector dim
    :param k: number of field
    :return:
    """
    linear_weight = tf.Variable(tf.truncated_normal(shape=[n, 1], dtype=tf.float32, mean=0, stddev=0.01), name='linear_weight')
    V = tf.Variable(tf.truncated_normal(shape=[n, k, p], dtype=tf.float32, mean=0, stddev=0.01), name="cross_weight")
    b = tf.Variable(tf.zeros([1], dtype=tf.float32), name="bias")
    w_last = tf.Variable(tf.truncated_normal([1, 1], dtype=tf.float32, mean=0, stddev=0.01), name="last_layer")
    b_last = tf.Variable(tf.constant(0.01, dtype=tf.float32), name="b_last")
    return linear_weight, V, b, w_last, b_last


def ffm(x_train, linear_w, x_field, V, b, n, k, w_last, b_last):
    """
    ffm model
    :param x_train:
    :param linear_w:
    :param x_field:
    :param V:
    :param b:
    :param n:
    :param k:
    :param w_last:
    :param b_last:
    :return:
    """
    linear_term = tf.add(
        b,
        tf.matmul(
            x_train,
            linear_w
        )
    )
    cross_term = tf.Variable(tf.zeros([1], dtype=tf.float32))
    for i in range(n):
        featureIndex1 = i
        featureField1 = int(x_field[i])
        for j in range(i+1, n):
            featureIndex2 = j
            featureField2 = int(x_field[j])
            LeftVec = tf.convert_to_tensor([[featureIndex1, featureField2, i] for i in range(k)])
            LeftVecAfterCut = tf.squeeze(tf.gather_nd(V, LeftVec))
            RightVec = tf.convert_to_tensor([[featureIndex2, featureField1, i] for i in range(k)])
            RightVecAfterCut = tf.squeeze(tf.gather_nd(V, RightVec))
            xi = tf.squeeze(tf.gather_nd(tf.transpose(x_train), [i]))
            xj = tf.squeeze(tf.gather_nd(tf.transpose(x_train), [j]))
            tempVal = tf.reduce_sum(
                tf.multiply(LeftVecAfterCut,
                            RightVecAfterCut)
            )
            product = tf.multiply(
                tf.transpose(xi),
                tf.transpose(xj)
            )
            temp = tf.multiply(tempVal, product)
            tf.assign(cross_term, tf.add(cross_term, temp))
    out = tf.add(linear_term, cross_term)
    output = tf.add(b_last, tf.matmul(out, w_last))
    y_hat = tf.nn.sigmoid(output)
    return y_hat


def loss(y_hat, y_train, linear_w, V, lambda_1=0.01, lambda_2=0.002):
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


def train(label, x, x_field, p=5, epochs=1000, learning_rate=0.0001):
    """
    train FFM
    :param label:
    :param x:
    :param x_field:
    :param p:
    :param epochs:
    :param learning_rate:
    :return:
    """
    m, n = x.shape
    k = int(np.max(x_field))
    x_train = tf.placeholder(shape=[None, n], dtype=tf.float32)
    y_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    linear_w, V, b, w_last, b_last = init_weight(n, p, k)
    y_hat = ffm(x_train, linear_w, x_field, V, b, n, k, w_last, b_last)
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
    label, x, x_field = load_data()
    train_loss_list, log_loss_list, y_output = train(label, x, x_field)
    y_ = [1 if data > 0.5 else 0 for data in y_output]
    pr = float(np.sum([1 if y_[i] == label[i] else 0 for i in range(len(label))])) / float(len(label))
    return train_loss_list, log_loss_list, y_output, pr


if __name__ == "__main__":
    train_loss_list, log_loss_list, y_output, pr = main()
    print("the precision is %s" % (str(pr)))
    print(y_output[:10])
    plt.plot(range(1000), train_loss_list)
    plt.show()
