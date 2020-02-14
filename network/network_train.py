import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. IN, nothing to do, has been dealt with!')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import pickle
import datetime
from keras.layers import Input
from keras.models import Model 
from keras.layers import Conv2D, MaxPooling2D, Activation 
from keras.layers import Reshape, Permute
from keras.layers import BatchNormalization, TimeDistributed, Dense, Dropout
from keras.layers import GRU, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from tensorflow.python.client import device_lib
from sklearn.metrics import roc_curve, auc


def create_model(filters, gru_units, dense_neurons, dropout):
    """
    # Arguments
        filters: number of filters in the convolutional layers.
        gru_units: number of gru units in the GRU layers.
        dense_neurons: number of neurons in the dense layers.
        dropout: neurons to drop out during training. Values of 0 to 1.

    # Returns
        Keras functional model which can then be compiled and fit.
    """
    inp = Input(shape=(259, 64, 1))
    c_1 = Conv2D(filters, (3,3), padding='same', activation='relu')(inp)
    mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
    c_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_1)
    mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
    c_3 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_2)
    mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)

    reshape_1 = Reshape((x_train.shape[-3], -1))(mp_3)
    rnn_1 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                              recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(reshape_1)
    rnn_2 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                              recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(rnn_1)
    
    dense_1  = TimeDistributed(Dense(dense_neurons, activation='relu'))(rnn_2)
    drop_1 = Dropout(rate=dropout)(dense_1)
    dense_2 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_1)
    drop_2 = Dropout(rate=dropout)(dense_2)
    dense_3 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_2)
    drop_3 = Dropout(rate=dropout)(dense_3)
    output = TimeDistributed(Dense(8, activation='sigmoid'))(drop_3)

    model = Model(inp, output)
    return model


def save_folder(date_time):
    """
    # Arguments
        date_time: Current time as per datetime.datetime.now()

    # Returns
        path to save model and related history and metrics.
    """
    date_now = str(date_time.date())
    time_now = str(date_time.time())
    sf = "saved_models/model_" + date_now + "_" + time_now + "_" + os.path.basename(__file__).split('.')[0]
    return sf


def create_save_folder(save_folder):
    """
    # Arguments
        save_folder: path for directory to save model and related history and metrics.

    # Output
        creates directory at save_folder location, if it does not exist already.
    """ 
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

        
def save_model(save_folder):
    """
    # Arguments
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves model and history.
    """ 
    model.save(save_folder + '/savedmodel' + '.h5')
    with open(save_folder + '/history.pickle', 'wb') as f:
        pickle.dump(model_fit.history, f)


def plot_accuracy(model_fit, save_folder):
    """
    # Arguments
        model_fit: model after training.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves plots of train vs validation accuracy at each training epoch.
    """ 
    train_acc = model_fit.history['binary_accuracy']
    val_acc = model_fit.history['val_binary_accuracy']
    epoch_axis = np.arange(1, len(train_acc) + 1)
    plt.title('Train vs Validation Accuracy')
    plt.plot(epoch_axis, train_acc, 'b', label='Train Acc')
    plt.plot(epoch_axis, val_acc,'r', label='Val Acc')
    plt.xlim([1, len(train_acc)])
    plt.xticks(np.arange(min(epoch_axis), max(epoch_axis) + 1, round((len(train_acc) / 10) + 0.5)))
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig(save_folder + '/accuracy.png')
    plt.show()
    plt.close()
    

def plot_loss(model_fit, save_folder):
    """
    # Arguments
        model_fit: model after training.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves plots of train vs validation loss at each training epoch.
    """ 
    train_loss = model_fit.history['loss']
    val_loss = model_fit.history['val_loss']
    epoch_axis = np.arange(1, len(train_loss) + 1)
    plt.title('Train vs Validation Loss')
    plt.plot(epoch_axis, train_loss, 'b', label='Train Loss')
    plt.plot(epoch_axis, val_loss,'r', label='Val Loss')
    plt.xlim([1, len(train_loss)])
    plt.xticks(np.arange(min(epoch_axis), max(epoch_axis) + 1, round((len(train_loss) / 10) + 0.5)))
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(save_folder + '/loss.png')
    plt.show()
    plt.close()


def plot_ROC(model, x_test, y_test, save_folder):
    """
    # Arguments
        model: model after training.
        x_test: inputs to the network for testing.
        y_test: actual outputs for testing.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves plots of ROC.
    """ 
    predicted = model.predict(x_test).ravel()
    actual = y_test.ravel()
    fpr, tpr, thresholds = roc_curve(actual, predicted, pos_label=None)
    roc_auc = auc(fpr, tpr)
    plt.title('Test ROC AUC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(save_folder + '/ROC.png')
    plt.show()
    plt.close()


def plot_class_ROC(model, x_test, y_test, save_folder):
    """
    # Arguments
        model: model after training.
        x_test: inputs to the network for testing.
        y_test: actual outputs for testing.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves plots of ROC per class.
    """ 
    class_names = ['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP']
    for i in range(len(class_names)):
        predicted = model.predict(x_test)[:,:,i].ravel()
        actual = y_test[:,:,i].ravel()
        fpr, tpr, thresholds = roc_curve(actual, predicted, pos_label=None)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic ' + class_names[i])
        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(save_folder + '/class_ROC_' + class_names[i] + '.png')
        plt.show()
        plt.close()


def save_arch(model, save_folder):
    """
    # Arguments
        model: model after training.
        save_folder: path for directory to save model and related history and metrics.

    # Output
        saves network architecture.
    """
    with open(save_folder + '/architecture.txt','w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


os.environ["CUDA_VISIBLE_DEVICES"]="1"  # select GPU
# IN tensorflow version problem: config = tf.ConfigProto()
#config = tf.compat.v1.ConfigProto
#tf.compat.v1.Session(config=config)  
# IN tf.Session(config=config)

# load train and validation datasets

# x_train = np.load('../datasets/x_train.npy', allow_pickle=True)
# x_val = np.load('../datasets/x_val.npy', allow_pickle=True)
# x_test = np.load('../datasets/x_test.npy', allow_pickle=True)
# y_train = np.load('../datasets/y_train.npy', allow_pickle=True)
# y_val = np.load('../datasets/y_val.npy', allow_pickle=True)
# y_test = np.load('../datasets/y_test.npy', allow_pickle=True)

x_train = np.expand_dims(np.stack(np.load('../datasets/x_train.npy', allow_pickle=True), axis=0), axis=-1)
x_val = np.expand_dims(np.stack(np.load('../datasets/x_val.npy', allow_pickle=True), axis=0), axis=-1)
x_test = np.expand_dims(np.stack(np.load('../datasets/x_test.npy', allow_pickle=True), axis=0), axis=-1)

y_train = np.stack(np.load('../datasets/y_train.npy', allow_pickle=True), axis=0)
y_val = np.stack(np.load('../datasets/y_val.npy', allow_pickle=True), axis=0)
y_test = np.stack(np.load('../datasets/y_test.npy', allow_pickle=True), axis=0)
# y_train = np.expand_dims(np.stack(np.load('../datasets/y_train.npy', allow_pickle=True), axis=0), axis=-1)
# y_val = np.expand_dims(np.stack(np.load('../datasets/y_val.npy', allow_pickle=True), axis=0), axis=-1)
# y_test = np.expand_dims(np.stack(np.load('../datasets/y_test.npy', allow_pickle=True), axis=0), axis=-1)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[2], x_val.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1], 1)
#  x shapes: (Bs, time_steps, mel_bands, 1)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[2], y_train.shape[1])
y_val = y_val.reshape(y_val.shape[0], y_val.shape[2], y_val.shape[1])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[2], y_test.shape[1])
#   y shapes: (BS, time_steps, classes(8))
# import pdb; pdb.set_trace()

model = create_model(filters=128, gru_units=128, dense_neurons=1024, dropout=0.5)
print(model.summary())
# print(model)
# import pdb; pdb.set_trace()
#adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, epsilon=0.1, decay=0.0, amsgrad=False, beta_2=0.999) # IN removing epsilon = None made it work the Valeu error regarding None )

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

epochs = 2500
batch_size = 128
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
# import pdb; pdb.set_trace()
model_fit = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(x_val, y_val), shuffle=True,
                      callbacks=[early_stopping, reduce_lr_plat])
date_time = datetime.datetime.now()
sf = save_folder(date_time)
create_save_folder(sf)
save_model(sf)
plot_accuracy(model_fit, sf)
plot_loss(model_fit, sf)
plot_ROC(model, x_val, y_val, sf)
plot_class_ROC(model, x_val, y_val, sf)
save_arch(model, sf)
