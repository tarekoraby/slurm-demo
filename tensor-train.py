import json
import tensorflow as tf

# Load CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), _ = cifar10.load_data()
x_train = x_train / 255.0  # Normalize data

# Define simplified model for CIFAR-10
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10)
])

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define callback for training progress
class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        metadata = {
            "epoch": epoch,
            "accuracy": logs['accuracy'],
            "loss": logs['loss']
        }
        print(json.dumps(metadata))

# Train model with callback
model.fit(x_train, y_train, epochs=20, verbose= 0, callbacks=[ProgressCallback()])
