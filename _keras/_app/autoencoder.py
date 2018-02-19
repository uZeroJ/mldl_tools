"""Simple Autoencoder tutorial from Keras Blog.
https://blog.keras.io/building-autoencoders-in-keras.html
"""

import logging

import numpy as np
from keras.datasets.mnist import load_data
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
from keras.layers import LSTM, RepeatVector, Lambda
from keras.models import Model
from keras.regularizers import l1
from keras.metrics import binary_crossentropy
from keras.callbacks import TensorBoard
import keras.backend as K

# Compression factor is 24.5
ENCODE_DIM = 32
# Mnist with 28 * 28
RAW_PIC_SIZE = 28
INPUT_DIM = RAW_PIC_SIZE ** 2
CBK_DIR = '/tmp/tf_log'

# Name of representation layer
REP_LAYER_NAME = 'rep'

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
                    level=logging.INFO)


def get_smpl_model(sparsity=False, deep=False, cnn=False, big_cnn=False):
    """
    Get shallow autoencoder encoder/decoder model and encoder as well as
    decoder respectively.
    :return: (auteencoder, encoder, decoder)
    """
    if big_cnn:
        ksiz = 32
    else:
        ksiz = 16

    def build_encoder():
        if cnn:
            inp = Input(shape=(RAW_PIC_SIZE, RAW_PIC_SIZE, 1))
            hidden = Conv2D(64, (3, 3),
                            activation='relu',
                            padding='same')(inp)
            hidden = MaxPool2D((2, 2), padding='same')(hidden)
            hidden = Conv2D(ksiz, (3, 3),
                            activation='relu',
                            padding='same')(hidden)
            hidden = MaxPool2D((2, 2), padding='same')(hidden)
            hidden = Conv2D(ksiz, (3, 3),
                            activation='relu',
                            padding='same')(hidden)

            representation = MaxPool2D((2, 2), padding='same',
                                       name=REP_LAYER_NAME)(hidden)
        else:
            inp = Input(shape=(INPUT_DIM,))
            if deep:
                representation = Dense(128, activation='relu')(inp)
                representation = Dense(64, activation='relu')(representation)
                representation = Dense(ENCODE_DIM,
                                       activation='relu',
                                       name=REP_LAYER_NAME)(representation)
            else:
                if sparsity:
                    reg = l1(10e-3)
                else:
                    reg = None
                representation = Dense(ENCODE_DIM,
                                       activation='relu',
                                       activity_regularizer=reg,
                                       name=REP_LAYER_NAME)(inp)
        return inp, representation

    def build_decoder(inp, dec_only=False, model=None):
        if dec_only:
            new_inp = Input(shape=inp._keras_shape[1:])
            # Get decoder layers from autoencoder as it will be trained. And
            # we need to share these layers but only change input.
            rep_layer = model.get_layer(REP_LAYER_NAME)
            rep_idx = model.layers.index(rep_layer)
            dec_layers = model.layers[rep_idx + 1:]
            tmp_inp = new_inp
            for layer in dec_layers:
                tmp_inp = layer(tmp_inp)
            out = tmp_inp
            return new_inp, out
        else:
            if cnn:

                up_hidden = Conv2D(ksiz, (3, 3),
                                   activation='relu',
                                   padding='same',
                                   name='up_hidden')(inp)
                up_hidden = UpSampling2D((2, 2))(up_hidden)
                up_hidden = Conv2D(ksiz, (3, 3), activation='relu',
                                   padding='same')(
                    up_hidden)
                up_hidden = UpSampling2D((2, 2))(up_hidden)
                up_hidden = Conv2D(64, (3, 3), activation='relu',
                                   padding='valid')(
                    up_hidden)
                up_hidden = UpSampling2D((2, 2), name='resampled')(up_hidden)

                out = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(
                    up_hidden)
            else:
                if deep:
                    out = Dense(64, activation='relu', name='up_hidden')(inp)
                    out = Dense(128, activation='relu')(out)
                    out = Dense(INPUT_DIM, activation='sigmoid')(out)
                else:
                    out = Dense(INPUT_DIM, activation='sigmoid')(inp)

            return out

    inp, representation = build_encoder()
    out = build_decoder(representation)
    autoencoder = Model(inp, out)
    print(autoencoder.summary())
    encoder = Model(inp, representation)

    # Should only pass in layer or create new layer in side the functions.
    # In order to generate new input layer accordingly just get it from the
    # given representation layer.
    # rep_inp = Input(shape=(ENCODE_DIM,))
    new_inp, out = build_decoder(representation,
                                 dec_only=True,
                                 model=autoencoder)
    # Reuse the last layer of autoencoder by switch to new inputs.
    decoder = Model(new_inp, out)

    return autoencoder, encoder, decoder,


def get_seq_model():
    """
    Get seq2seq encoder-decoder model for autoencoder. By repeating the last
    output of the sequence hidden state and put it as input to each sequence
    step of decoder.
    For this simple example, just take row of MNIST data as timestep and the
    column as embedding, which is more or less like a row-based scanner.
    :return: models of (autoencoder, encoder, decoder)
    """
    # Use 128 states to represent the encoded data.
    SEQ_REP_SIZE = 128
    REPEAT_LAYER_NAME = 'repeat_layer'
    # encoder
    inp = Input(shape=(RAW_PIC_SIZE, RAW_PIC_SIZE))
    representation = LSTM(SEQ_REP_SIZE, name=REP_LAYER_NAME)(inp)

    # Use 28 as timestep which is stoed in RAW_PIC_SIZE
    encoded = RepeatVector(RAW_PIC_SIZE, name=REPEAT_LAYER_NAME)(representation)
    out = LSTM(RAW_PIC_SIZE,
               # activation='sigmoid',
               return_sequences=True)(encoded)

    seq_autoencoder = Model(inp, out)
    seq_encoder = Model(inp, representation)

    repeat_layer = seq_autoencoder.get_layer(REPEAT_LAYER_NAME)
    new_inp = Input(shape=repeat_layer._keras_shape)
    new_out = seq_autoencoder.layers[-1](new_inp)
    seq_decoder = Model(new_inp, new_out)

    return seq_autoencoder, seq_encoder, seq_decoder


def get_vae_model(batch_size=32):
    """
    Get Variational AutoEncoder model.
    :return: models of (autoencoder,
    """
    INTER_HSIZE = 128
    LATENT_HSIZE = 2
    ALPHA = 0.5

    x = Input(shape=(INPUT_DIM,))
    inter = Dense(INTER_HSIZE, activation='relu')(x)

    z_mean = Dense(LATENT_HSIZE)(inter)
    z_log_var = Dense(LATENT_HSIZE)(inter)

    def sampling(arg):
        mean, var = arg
        rn = K.random_normal(shape=(batch_size, LATENT_HSIZE),
                             mean=0., stddev=1.)
        z = mean + K.exp(var / 2) * rn
        return z

    z = Lambda(sampling, output_shape=(INPUT_DIM,))([z_mean, z_log_var])

    # Decode from sampled data as it is generated from encoded latent
    # distribution.
    dec_inter = Dense(INTER_HSIZE, activation='relu', name='dec_inter')(z)
    # Reconstruct from mean value.
    dec_mean = Dense(INPUT_DIM, activation='simgoid', name='dec_mean')(
        dec_inter)

    vae_model = Model(x, dec_mean)
    x_loss = INPUT_DIM * binary_crossentropy(x, dec_mean)
    kl_loss = - ALPHA * K.sum(1 + z_log_var
                              - K.square(z_mean)
                              - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(x_loss + kl_loss)
    vae_model.compile(optimizer='rmsprop', loss=vae_loss)
    # vae_model.summary()

    vae_enc = Model(x, z_mean)

    # The decoder
    dec_inp = Input(shape=(LATENT_HSIZE,))
    new_dec_inter = vae_model.get_layer('dec_inter')(dec_inp)
    new_dec_mean = vae_model.get_layer('dec_mean')(new_dec_inter)
    vae_gen = Model(dec_inp, new_dec_mean)

    return vae_model, vae_enc, vae_gen


def get_mnist_data(cnn=False, noisy=False):
    """
    Get train/test data from mnist for autoencoder.
    :return: only (x_train, X_test) as we don't care about labels.
    """
    (X_train, _), (X_test, _) = load_data()
    # Normalize data
    X_train = X_train.astype(np.float32) / 255.
    X_test = X_test.astype(np.float32) / 255.

    if cnn:
        # Add channel axis with 1, then the shape should be (?, 28, 28, 1)
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
    else:
        # flatten data.
        X_train = X_train.reshape(
            (X_train.shape[0], np.prod(X_train.shape[1:])))
        X_test = X_test.reshape((X_test.shape[0], np.prod(X_test.shape[1:])))

    if noisy:
        NOISY_FACTOR = 0.5
        X_train = X_train + NOISY_FACTOR * np.random.normal(loc=0,
                                                            scale=1.0,
                                                            size=X_train.shape)
        X_test = X_test + NOISY_FACTOR * np.random.normal(loc=0,
                                                          scale=1.0,
                                                          size=X_test.shape)
        # Need to clip to (0, 1)
        X_train = np.clip(X_train, 0, 1)
        X_test = np.clip(X_test, 0, 1)

    logging.info('Train shape: {}, Test shape: {}'.format(X_train.shape,
                                                          X_test.shape))

    return X_train, X_test


def train_autoencoder(model, X_train, X_test, noisy=False,
                      epoch=50, batch_size=256, call_back_dir=None,
                      vae=False):
    if vae:
        # Self-defined loss function
        # def vae_loss(alpha=0.5):
        # TODO, Why product dimension here?
        # x_loss = INPUT_DIM * binary_crossentropy(vae.input,
        #                                          vae.output)
        # kl_loss = - alpha *
        raise NotImplementedError('Need to split loss function from the '
                                  'defined layers!')
    else:
        # train_data, test_data = X_train, X_test
        if noisy:
            train_lbl, test_lbl = get_mnist_data(noisy=noisy)
        else:
            train_lbl, test_lbl = X_train, X_test

        if call_back_dir:
            cbks = [TensorBoard(log_dir=call_back_dir)]
        else:
            cbks = None

        model.compile('adadelta', 'binary_crossentropy')
        model.fit(X_train, train_lbl,
                  epochs=epoch,
                  batch_size=batch_size,
                  shuffle=True,
                  validation_data=(X_test, test_lbl),
                  callbacks=cbks)


def predict(model, X_test):
    if isinstance(model, list):
        enc_model, dec_model = model
        encoded_data = enc_model.predict(X_test)
        decoded_data = dec_model.predict(encoded_data)
    else:
        decoded_data = model.predict(X_test)

    return decoded_data


def plot_decoded(raw_input, decoded):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 4))

    DISPLAY_NUM = 10
    for idx in range(DISPLAY_NUM):
        ax = plt.subplot(2, DISPLAY_NUM, idx + 1)
        plt.imshow(raw_input[idx].reshape((RAW_PIC_SIZE,
                                           RAW_PIC_SIZE)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, DISPLAY_NUM, DISPLAY_NUM + idx + 1)
        plt.imshow(decoded[idx].reshape((RAW_PIC_SIZE,
                                         RAW_PIC_SIZE)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


if __name__ == '__main__':
    get_vae_model()
    use_cnn = True
    use_sp = False
    use_deep = False
    epoch = 100
    auto_enc, enc, dec = get_smpl_model(sparsity=use_sp,
                                        deep=use_deep,
                                        cnn=use_cnn)
    X_train, X_test = get_mnist_data(cnn=use_cnn)

    train_autoencoder(auto_enc, X_train, X_test, epoch=epoch)
    decoded_data = predict([enc, dec], X_test)
    plot_decoded(X_test, decoded_data)
