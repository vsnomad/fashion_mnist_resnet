import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import load_dataset
from mnist_fashion.classifier import convolutional_model
from mnist_fashion.classifier import resnet

import pandas as pd

dataset = load_dataset("fashion_mnist").with_format("tf")

X_train = tf.expand_dims(dataset['train']['image'], axis=-1)
X_test = tf.expand_dims(dataset['test']['image'], axis=-1)
Y_train = tf.one_hot(dataset['train']['label'], depth=10)
Y_test = tf.one_hot(dataset['test']['label'], depth=10)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)

# model = convolutional_model((28, 28, 1))
model = resnet((28, 28, 1), 10)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(train_dataset, epochs=15, validation_data=test_dataset)

df_loss_acc = pd.DataFrame(history.history)
df_loss = df_loss_acc[['loss', 'val_loss']]
df_loss.rename(columns={'loss': 'train', 'val_loss': 'validation'}, inplace=True)
df_acc = df_loss_acc[['accuracy', 'val_accuracy']]
df_acc.rename(columns={'accuracy': 'train', 'val_accuracy': 'validation'}, inplace=True)
df_loss.plot(title='Model loss', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Loss')
df_acc.plot(title='Model Accuracy', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Accuracy')

df_loss_acc = pd.DataFrame(history.history)
df_loss = df_loss_acc[['loss', 'val_loss']]
df_loss.rename(columns={'loss': 'train', 'val_loss': 'validation'}, inplace=True)
df_acc = df_loss_acc[['accuracy', 'val_accuracy']]
df_acc.rename(columns={'accuracy': 'train', 'val_accuracy': 'validation'}, inplace=True)
df_loss.plot(title='Model loss', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Loss')
plt.show()
df_acc.plot(title='Model Accuracy', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Accuracy')
plt.show()
