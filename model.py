import tensorflow as tf


class EmbeddingNet(tf.keras.Model):
    def __init__(self,
                 dim_dense_1: int = 8,
                 dim_dense_2: int = 4,
                 dim_embedding: int = 2,
                 ):
        super(EmbeddingNet, self).__init__()
        self.layer1 = tf.keras.layers.Dense(dim_dense_1, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(dim_dense_2, activation=tf.nn.relu)
        self.out_layer = tf.keras.layers.Dense(dim_embedding)

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.out_layer(x)
