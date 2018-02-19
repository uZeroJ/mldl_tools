"""
Simple Examples of how to building Keras model.
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical
from keras.layers import Dropout


def bin_classifier():
    inp = Input(shape=(1000,))
    out = Dense(1, activation='sigmoid')(inp)

    model = Model(inp, out)
    model.compile('adam', 'binary_crossentropy', metrics=['acc'])

    inputs = np.random.random((300, 1000))
    labels = np.random.randint(2, size=(300, 1))

    model.fit(inputs, labels, epochs=10, batch_size=32)


def multi_classifier():
    inp = Input(shape=(1000,))
    out = Dense(10, activation='softmax')(inp)

    model = Model(inp, out)
    model.compile('adam', 'categorical_crossentropy', metrics=['acc'])

    inputs = np.random.random((1000, 1000))
    labels = to_categorical(np.random.randint(10, size=(1000, 1)))

    model.fit(inputs, labels, epochs=10, batch_size=32)


def simple_MLP():
    inp = Input(shape=(1000,))
    h1 = Dense(512, activation='relu')(inp)
    drp1 = Dropout(0.5)(h1)
    h2 = Dense(128, activation='relu')(drp1)
    drp2 = Dropout(0.5)(h2)
    out = Dense(10, activation='softmax')(drp2)

    model = Model(inp, out)
    model.compile('rmsprop', 'categorical_crossentropy', metrics=['acc'])

    X_train = np.random.random((1000, 1000))
    y_train = to_categorical(np.random.randint(10, size=(1000, 1)))

    X_test = np.random.random((200, 1000))
    y_test = to_categorical(np.random.randint(10, size=(200, 1)))

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    print('loss, accuracy:', model.evaluate(X_test, y_test, batch_size=32))


if __name__ == '__main__':
    # bin_classifier()
    # multi_classifier()
    simple_MLP()
