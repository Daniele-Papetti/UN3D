import pickle
import numpy as np
from math import ceil
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping


def build_ae(max_signal_length):
    input_layer = Input(shape=(max_signal_length, 1))
    # encoder
    conv_encoded1 = Conv1D(16, kernel_size=3, padding='same', activation='relu')(input_layer)
    conv_encoded2 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(conv_encoded1)
    max_pooling2 = MaxPooling1D(pool_size=2, padding='same')(conv_encoded2)
    conv_encoded3 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(max_pooling2)
    max_pooling3 = MaxPooling1D(pool_size=2, padding='same')(conv_encoded3)
    max_pooling3 = MaxPooling1D(pool_size=2, padding='same')(max_pooling3)
    conv_encoded4 = Conv1D(16, kernel_size=3, padding='same')(max_pooling3)
    bottleneck = MaxPooling1D(pool_size=2, padding='same')(conv_encoded4)
    # decoder
    depooling4 = UpSampling1D(size=2)(bottleneck)
    conv_encoded4 = Conv1D(16, kernel_size=3, padding='same')(depooling4)
    depooling3 = UpSampling1D(size=2)(conv_encoded4)
    conv_decoded3 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(depooling3)
    depooling2 = UpSampling1D(size=2)(conv_decoded3)
    conv_decoded2 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(depooling2)
    depooling1 = UpSampling1D(size=2)(conv_decoded2)
    conv_decoded1 = Conv1D(16, kernel_size=3, padding='same', activation='relu')(depooling1)
    output_layer = Conv1D(1, kernel_size=3, padding='same', activation='sigmoid')(conv_decoded1)
    output_layer = Cropping1D((0, output_layer.shape[1] - max_signal_length))(output_layer)

    ae = Model(input_layer, output_layer)
    encoder = Model(input_layer, bottleneck)

    ae.compile(optimizer='adam', loss="mse",
               metrics=['accuracy'])

    return ae, encoder


def signals_embedding(x, dump_file, dump_ae=None, dump_encoder=None, dump_train_signals=None):
    '''

    Trains the autoencoder with the whole dataset. Then, the trained autoencoder
    is used to create the embeddings of every element of the dataset.

    Parameters
    ----------
    x : list
        list of signals.

    dump_file: String
        the file to save the embeddings created by the autoencoder.

    dump_ae: String
        file to dump the autoencoder.

    dump_encoder: String
        file to dump the encoder.

    dump_train_signals: String
        file to dump the pre-processed signals.

    Returns
    -------
    history : keras history
        keras history of the autoencoder training phase.

    '''

    print(len(x))
    # Considering the first half of the signlas
    x_BC = list()
    for s in x:
        m = ceil(len(s) / 2)
        x_BC.append(s[:m + 1])
    max_signal_length = max([len(s) for s in x_BC])
    x_train = list()
    # padding the signals
    for s in x_BC:
        e = s
        while len(e) < max_signal_length:
            e = np.concatenate([e, e], axis = 0)
        e = e[:max_signal_length]
        x_train.append(e)
    print(max_signal_length)
  
    maxs = [np.max(s) for s in x]
    max_signal_value = max(maxs)
    print(max_signal_value)
    x_train = np.array([s / max_signal_value for s in x_train])
    x_train = np.reshape(x_train, (len(x_train), max_signal_length, 1))
    print(x_train.shape)

    es = EarlyStopping(monitor='val_loss', mode='min',
                       verbose=1, patience=5, min_delta=0.01,
                       restore_best_weights=True)

    ae, encoder = build_ae(max_signal_length)
  
    # train autoencoder
    print("Training model")
    history = ae.fit(x_train, x_train,
                     validation_split=0.2, epochs=2000,
                     batch_size=64, verbose=2, shuffle=True,
                     callbacks=[es], use_multiprocessing=False)

    # predictions
    preds = encoder.predict(x_train, batch_size=32,
                            use_multiprocessing=True, verbose=False)
    with open(dump_file, 'wb') as f:
        pickle.dump(preds, f)

    if dump_ae:
        ae.save(dump_ae)
    if dump_encoder:
        encoder.save(dump_encoder)
    if dump_train_signals:
        with open(dump_train_signals, 'wb') as f:
            pickle.dump(x_train, f)

    return history
