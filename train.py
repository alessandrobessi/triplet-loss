import tensorflow as tf
from losses import triplet_semihard_loss

train_loss = tf.keras.metrics.Mean(name='train_loss')


def loss_fn(model: tf.keras.Model, inputs: tf.Tensor, labels: tf.Tensor) -> float:
    return triplet_semihard_loss(labels, model(inputs))


@tf.function
def train_step(model: tf.keras.Model,
               x_batch: tf.Tensor,
               y_batch: tf.Tensor,
               optimizer: tf.keras.optimizers) -> tf.keras.metrics.Mean:
    with tf.GradientTape() as tape:
        loss = loss_fn(model, x_batch, y_batch)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return train_loss(loss)
