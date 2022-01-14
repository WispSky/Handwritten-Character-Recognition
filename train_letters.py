import keras
from keras.models import Sequential
import tensorflow as tf
from extra_keras_datasets import emnist

# load dataset
(x_train, y_train), (x_test, y_test) = emnist.load_data(type='letters')

# print(x_train.shape, y_train.shape)

# resize shape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# convert vectors to binary matrices
y_train = keras.utils.np_utils.to_categorical(y_train-1, 26)
y_test = keras.utils.np_utils.to_categorical(y_test-1, 26)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# modifiable training variables
batch_size = 128
num_classes = 26
epochs = 100

# create model
model = tf.keras.Sequential([ 
    tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
# compile model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# callbacks for improving accuracy of model
# save checkpoint models throughout training whenever there is decent improvement
MCP = tf.keras.callbacks.ModelCheckpoint('Best_points.h5', verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')
# stop training early if there is little (or no) improvement
ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, verbose=0, restore_best_weights = True, patience=3, mode='max')
# reduces learning rate if no improvement is seen
RLP = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=0.0001)
# list of callbacks
cbs = [MCP, ES, RLP]

# train the model
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=cbs)
print('The model is finished training.')

# print results
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

# save model
model.save('emnist.h5')
print('Saving model to file \'emnist.h5\'')