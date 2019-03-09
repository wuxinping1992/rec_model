import tensorflow as tf
from build_data import load_data
import numpy as np


class Args():
    feature_sizes = 100
    filed_size = 15
    embedding_size = 256
    deep_layers = [512, 256, 128]
    epoch = 3
    batch_size = 64
    learning_rate = 1.0
    l2_reg_rate = 0.01
    checkpoint_dir = './model/'
    is_traing = True
    deep_activation = tf.nn.relu


class DeepFM:
    def __init__(self, args):
        self.feature_sizes = args.feature_sizes
        self.field_size = args.filed_size
        self.embedding_size = args.embedding_size
        self.deep_layers = args.deep_layers
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.l2_reg_rate = args.l2_reg_rate
        self.deep_activation = tf.nn.relu
        self.weight = dict()
        self.checkpoint_dir = args.checkpoint_dir
        self.build_model()

    def build_model(self):
        self.feat_index = tf.placeholder(shape=[None, None], dtype=tf.int32, name='feat_index')
        self.feat_value = tf.placeholder(shape=[None, None], dtype=tf.float32, name='feat_value')
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y')

        # feature vector
        # V
        self.weight["feature_weight"] = tf.Variable(
            tf.random_normal(shape=[self.feature_sizes, self.embedding_size], mean=0.0, stddev=0.01),
            name='feature_weight'
        )
        # linear term weight
        self.weight["feature_first"] = tf.Variable(
            tf.random_normal(shape=[self.feature_sizes, 1], mean=0.0, stddev=0.01),
            name="feature_first"
        )

        # deep weight and bias
        input_size = self.embedding_size * self.field_size
        init_method = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        self.weight["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=init_method, size=(input_size, self.deep_layers[0])),
            name='layer_0', dtype=np.float32
        )
        self.weight["bias_0"] = tf.Variable(
            np.random.normal(loc=0, scale=init_method, size=(1, self.deep_layers[0])),
            name='bias_0', dtype=np.float32
        )
        num_layers = len(self.deep_layers)
        if num_layers != 1:
            for i in range(1, num_layers):
                init_method = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
                self.weight["layer_" + str(i)] = tf.Variable(
                    np.random.normal(loc=0., scale=init_method, size=(self.deep_layers[i-1], self.deep_layers[i])),
                    name='layer_%s' % i, dtype=np.float32
                )
                self.weight["bias_" + str(i)] = tf.Variable(
                    np.random.normal(loc=0., scale=init_method, size=(1, self.deep_layers[i])),
                    name='bias_%s' % i, dtype=np.float32
                )

        # deep output_size + first term out_size + second term output_size
        last_layer_size = self.deep_layers[-1] + self.field_size + self.embedding_size
        init_method = np.sqrt(2.0 /(last_layer_size + 1))
        self.weight["last_layer"] = tf.Variable(
            np.random.normal(loc=0, scale=init_method, size=(last_layer_size,1)),
            dtype=np.float32, name='last_layer'
        )
        self.weight["last_bias"] = tf.Variable(
            tf.constant(0.01), dtype=tf.float32, name='last_bias'
        )

        #embedding part
        self.embedding_weight = tf.nn.embedding_lookup(self.weight["feature_weight"], self.feat_index)
        self.embedding_part = tf.multiply(
            self.embedding_weight,
            tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
        )
        #first order
        self.embedding_first = tf.nn.embedding_lookup(self.weight["feature_first"], self.feat_index)
        self.embedding_first = tf.multiply(
            self.embedding_first,
            tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
        )
        self.first_order = tf.reduce_sum(self.embedding_first, axis=2)

        #second order
        self.sum_second_order = tf.reduce_sum(self.embedding_part, axis=1)
        self.sum_second_order_square = tf.square(self.sum_second_order)
        self.square_sum_second = tf.square(self.embedding_part)
        self.square_sum_second_sum = tf.reduce_sum(self.square_sum_second, axis=1)
        self.second_order = 0.5 * tf.subtract(self.sum_second_order_square, self.square_sum_second_sum)

        # fm part
        self.fm_part = tf.concat([self.first_order, self.second_order], axis=1)

        # deep part
        self.deep_embedding = tf.reshape(self.embedding_part, shape=[-1, self.field_size * self.embedding_size])

        for i in range(0, len(self.deep_layers)):
            self.deep_embedding = tf.add(
                tf.matmul(self.deep_embedding, self.weight["layer_" + str(i)]),
                self.weight["bias_" + str(i)]
            )
            self.deep_embedding = self.deep_activation(self.deep_embedding)

        # concat
        din_all = tf.concat([self.fm_part, self.deep_embedding], axis=1)
        self.out = tf.add(
            tf.matmul(din_all, self.weight["last_layer"]),
            self.weight["last_bias"]
        )
        # loss
        self.out = tf.nn.sigmoid(self.out)
        self.loss = -tf.reduce_mean(
            self.y * tf.log(self.out + 1e-24) + (1 - self.y) * tf.log(1 - self.out + 1e-24)
        )
        self.loss += self.l2_reg_rate * tf.reduce_mean(tf.pow(self.weight["last_layer"], 2))
        for i in range(len(self.deep_layers)):
            self.loss += self.l2_reg_rate * tf.reduce_mean(tf.pow(self.weight["layer_" + str(i)], 2))
        print(self.loss)
        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()
        print(trainable_params)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, feat_index, feat_value, label):
        train_loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.feat_value: feat_value,
            self.feat_index: feat_index,
            self.y: label
        })
        return train_loss

    def predict(self, sess, feat_index, feat_value):
        result = sess.run([self.out], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], y[start:end]


if __name__ == "__main__":
    args = Args()
    data = load_data()
    args.feature_sizes = data['feat_dim']
    args.filed_size = len(data['Xi'][0])
    args.is_traing = True
    with tf.Session() as sess:
        DeepFmModel = DeepFM(args)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        cnt = int(len(data['y_train']) / args.batch_size)
        print("time all: %s" % cnt)
        if args.is_traing:
            for i in range(args.epoch):
                for j in range(cnt):
                    X_index, X_value, y_train = get_batch(data['Xi'], data['Xv'], data["y_train"], args.batch_size, j)
                    train_loss_ = DeepFmModel.train(sess, X_index, X_value, y_train)
                    if j % 100 == 0:
                        print("the times of training is %d, and the loss is %s" % (j, train_loss_))
                        DeepFmModel.save(sess, args.checkpoint_dir)
        else:
            DeepFmModel.restore(sess, args.checkpoint_dir)
            for j in range(0, cnt):
                X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                result = DeepFmModel.predict(sess, X_index, X_value)
                print(result)
