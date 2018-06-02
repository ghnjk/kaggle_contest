import tensorflow as tf
import numpy as np


class TitanicModel(object):

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

        # tf inputs
        self.tf_age = None
        self.tf_sex = None
        self.tf_embarked = None
        self.tf_pclass = None
        self.tf_fare = None
        self.tf_sibsp = None
        self.tf_parch = None
        self.tf_survived = None
        # tf processing
        self.tf_physical_feature = None
        self.tf_economic_feature = None
        self.tf_relation_feature = None
        # tf outputs
        self.tf_predict_survived_probs = None
        self.tf_loss = None
        self.tf_train_op = None

        self.sess = tf.Session()
        self._build_net()
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, max_epoch=30000, batch_size=64):
        for epoch in range(max_epoch):
            feature, label = data.next_train_batch(batch_size)
            feed_dict = {
                self.tf_age: feature[:, data.feature_index["Age"]],
                self.tf_sex: feature[:, data.feature_index["Sex"]],
                self.tf_embarked: feature[:, data.feature_index["Embarked"]],
                self.tf_pclass: feature[:, data.feature_index["Pclass"]],
                self.tf_fare: feature[:, data.feature_index["Fare"]],
                self.tf_sibsp: feature[:, data.feature_index["SibSp"]],
                self.tf_parch: feature[:, data.feature_index["Parch"]],
                self.tf_survived: label
            }
            _, loss, prediction = self.sess.run([
                self.tf_train_op,
                self.tf_loss,
                self.tf_predict_survived_probs
            ], feed_dict=feed_dict)
            real = np.argmax(label, axis=1)
            prediction = np.argmax(prediction, axis=1)
            succ_count = np.sum(real == prediction)
            print("epoch", epoch, "loss:", loss, "accuracy: ", succ_count / float(batch_size))

    def predict(self, data):
        feature = data.test_feature
        feed_dict = {
            self.tf_age: feature[:, data.feature_index["Age"]],
            self.tf_sex: feature[:, data.feature_index["Sex"]],
            self.tf_embarked: feature[:, data.feature_index["Embarked"]],
            self.tf_pclass: feature[:, data.feature_index["Pclass"]],
            self.tf_fare: feature[:, data.feature_index["Fare"]],
            self.tf_sibsp: feature[:, data.feature_index["SibSp"]],
            self.tf_parch: feature[:, data.feature_index["Parch"]],
        }
        prediction = self.sess.run(self.tf_predict_survived_probs, feed_dict=feed_dict)
        prediction = np.argmax(prediction, axis=1)
        return prediction

    def _build_net(self):
        self.tf_survived = tf.placeholder(tf.float32, shape=[None, 2], name="suvived")
        with tf.variable_scope("physical_feature"):
            with tf.variable_scope("inputs"):
                self.tf_age = tf.placeholder(tf.float32, shape=[None, ], name="age")
                self.tf_sex = tf.placeholder(tf.float32, shape=[None, ], name="sex")
                self.tf_embarked = tf.placeholder(tf.float32, shape=[None, ], name="embarked")
                batch_size = tf.shape(self.tf_age)[0]
            physical_feature = tf.concat([
                tf.reshape(self.tf_age, [batch_size, 1]),
                tf.reshape(self.tf_sex, [batch_size, 1]),
                tf.reshape(self.tf_embarked, [batch_size, 1])
            ], axis=1, name="stack_all_physical_inputs")
            physical_feature = tf.layers.dense(
                inputs=physical_feature,
                units=64,
                activation=tf.nn.tanh,
                name="physical_value"
            )
            self.tf_physical_feature = physical_feature

        with tf.variable_scope("economic_feature"):
            with tf.variable_scope("inputs"):
                self.tf_pclass = tf.placeholder(tf.float32, shape=[None, ], name="pclass")
                self.tf_fare = tf.placeholder(tf.float32, shape=[None, ], name="fare")
            economic_feature = tf.concat([
                tf.reshape(self.tf_pclass, [batch_size, 1]),
                tf.reshape(self.tf_fare, [batch_size, 1])
            ], axis=1, name="stack_all_economic_inputs")
            economic_feature = tf.layers.dense(
                inputs=economic_feature,
                units=64,
                activation=tf.nn.tanh,
                name="economic_value"
            )
            self.tf_economic_feature = economic_feature

        with tf.variable_scope("relation_feature"):
            with tf.variable_scope("inputs"):
                self.tf_sibsp = tf.placeholder(tf.float32, shape=[None, ], name="sibsp")
                self.tf_parch = tf.placeholder(tf.float32, shape=[None, ], name="parch")
            relation_feature = tf.concat([
                tf.reshape(self.tf_sibsp, [batch_size, 1]),
                tf.reshape(self.tf_parch, [batch_size, 1])
            ], axis=1, name="stack_all_relation_inputs")
            relation_feature = tf.layers.dense(
                inputs=relation_feature,
                units=64,
                activation=tf.nn.tanh,
                name="relation_value"
            )
            self.tf_relation_feature = relation_feature

        with tf.variable_scope("survived_probability"):
            prediction = tf.concat([
                self.tf_physical_feature,
                self.tf_economic_feature,
                self.tf_relation_feature
            ], axis=1, name="concat_all_feature_values")
            prediction = tf.layers.dense(
                inputs=prediction,
                units=2,
                activation=tf.nn.softmax,
                name="survived_probs"
            )
            self.tf_predict_survived_probs = prediction

        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(
                - tf.reduce_sum(
                    self.tf_survived * tf.log(self.tf_predict_survived_probs + 1e-8),
                    reduction_indices=1
                )
            )
            self.tf_loss = loss

        with tf.variable_scope("train"):
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.tf_loss)
            self.tf_train_op = train_op
