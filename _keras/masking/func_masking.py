"""
Try to understand the working of Masking Layer.
https://github.com/keras-team/keras/issues/3086
"""

import numpy as np
from keras.layers import Input, Masking, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K


def one_d_work(msk_val=0):
    """
    Masking layer should not work with 1-D inputs.
    :return: None.
    """

    inputs = np.arange(1, 9).reshape((1, 8))
    print(inputs)
    inputs = np.where(inputs > 5, msk_val, inputs)
    print(inputs)
    # inputs[inputs > 5] = msk_val

    inp = Input(shape=(8,))
    msk = Masking(mask_value=msk_val)(inp)
    out = Activation('softmax')(msk)

    model = Model(inp, out)

    print(model.predict(inputs))


def two_d_work(msk_val=0):
    """
    Masking on 2-D inputs and seems only work on time step dimension.
    And just indicate this with 'leaky relu' activation. And should (
    positive, small zero, negative).
    :param msk_val: what value represent masking.
    :return: None.
    """
    inputs = np.array([[[2, 3, msk_val],
                       [msk_val] * 3,
                       [-1, -2, msk_val]]])
    print(inputs, inputs.shape)

    inp = Input(shape=(3, 3))
    msk = Masking(mask_value=msk_val)(inp)
    out = LeakyReLU()(msk)
    # out = Activation('relu')(msk)

    model = Model(inp, out)
    print(model.predict(inputs))


if __name__ == '__main__':
    # one_d_work(-1)
    two_d_work(10)
