import tensorflow as tf
from dataset import TripletLossDataset
from train import train_step
from model import EmbeddingNet

num_steps = 1000
learning_rate = 0.001
display_step = 100

data = TripletLossDataset(num_clusters=4, num_examples=256, batch_size=16)
data.build_dataset()

model = EmbeddingNet()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = None

for step in range(num_steps):
    for batch in data.dataset:
        x_batch = tf.cast(batch[0], dtype=tf.float32)
        y_batch = tf.cast(batch[1], dtype=tf.float32)
        loss = train_step(model, x_batch, y_batch, optimizer)
    if step % display_step == 0:
        print(f"Step: {step:04d}\tLoss: {loss.numpy():.4f}")
