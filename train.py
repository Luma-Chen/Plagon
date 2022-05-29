from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, BatchNormalization, Activation
from keras.optimizers import RMSprop
from keras import backend as K

import tensorflow as tf

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib as mpl 
import matplotlib.pyplot as plt 
plt.switch_backend('agg')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json

import argparse, os, pickle, re, random, sys
from tensorflow.python.keras.optimizers import Adadelta

import keras

from numpy.random import seed

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, datasets_dir, subdir, list_IDs, labels, batch_size=32, dim=(2400,),
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels

        self.dataset_dir = os.path.join(datasets_dir, subdir)
        self.list_IDs = list_IDs
        self.n_examples = len(list_IDs)

        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_examples / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_examples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        assert(len(list_IDs_temp) == self.batch_size)
        assert(y.shape[0] == self.batch_size)

        return X, y

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = [np.empty((self.batch_size, *self.dim)), np.empty((self.batch_size, *self.dim))]
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        # see https://stackoverflow.com/questions/53978295/error-using-fit-generator-with-a-siamese-network
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            filename = os.path.join(self.dataset_dir, 'ID{:010d}.npy'.format(ID))
            vecs = np.load(filename)
            vec1 = vecs[0:2400]
            vec2 = vecs[2400:]
            assert(vec1.shape == vec2.shape)
            X[0][i] = vec1
            X[1][i] = vec2

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def pearson_correlation(y_true, y_pred):
    # Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
    fs_pred = y_pred - K.mean(y_pred)
    fs_true = y_true - K.mean(y_true)
    covariance = K.mean(fs_true * fs_pred)
    
    stdv_true = K.std(y_true)
    stdv_pred = K.std(y_pred)
    
    return covariance / (stdv_true * stdv_pred)

def create_base_network_with_BN(emb_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=emb_shape)
    x = Dense(128, use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(128, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(128, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return Model(input, x)

def create_base_network(emb_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    n_neurons = 128

    input = Input(shape=emb_shape)
    x = Dense(n_neurons, activation='relu', kernel_initializer='he_normal')(input)
    x = Dropout(0.5)(x)
    x = Dense(n_neurons, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_neurons, activation='relu', kernel_initializer='he_normal')(x)
    #x = Dropout(0.5)(x) #NEW
    return Model(input, x)

def plot_losses(history):
    plt.plot(history.history['pearson_correlation'])
    plt.plot(history.history['val_pearson_correlation'])
    plt.title('model pearson_correlation')
    plt.ylabel('pearson_correlation')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show()
    plt.savefig('pearson_correlation.png')

    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show()
    plt.savefig('loss.png')

def get_partition(datasets_dir, subdir):
    subdir = os.path.join(datasets_dir, subdir)
    filelist = os.listdir(subdir)
    filelist.sort()
    filelist.pop() # remove the last item (i.e., file name 'labels.npy')

    ids = [re.findall(r'\d+', filename) for filename in filelist]
    ids = np.asarray([int(x[0]) for x in ids])

    labels_filename = os.path.join(datasets_dir, subdir, 'labels.npy')
    labels = np.load(labels_filename)    

    return ids, labels 

def main():
    parser = argparse.ArgumentParser(description='Train the siamese net.')
    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=40, 
        required=False)
    parser.add_argument(
        "--datasets_dir",
        help = 'source directory of train/val/test datasets')
    parser.add_argument(
        "--model_dir",
        help = 'destination directory for the resulting model')
    parser.add_argument('--shuffle', 
        dest='shuffle', 
        default=False, 
        help = 'Activate shuffle mode on the training dataset',
        action='store_true')
    
    args = parser.parse_args()

    num_classes = 2
    epochs = args.max_epochs

    input_shape = (2400,)

    # network definition
    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    l1_norm = lambda x: 1 - K.abs(x[0] - x[1])

    distance = Lambda(function=l1_norm,
                      output_shape=lambda x: x[0])([processed_a, processed_b])

    predictions = Dense(2, activation='sigmoid', name='Similarity_layer')(distance)
    
    model = Model([input_a, input_b], predictions)

    # train
#    model.compile(loss=contrastive_loss, optimizer=RMSprop(), metrics=[accuracy])
#    optimizer = RMSprop(0.000005)
    optimizer = keras.optimizers.Adam(lr=0.000005)
    model.compile(loss = 'mse', optimizer = optimizer, metrics=[pearson_correlation])

    patience = int(0.2 * epochs)
    print('Value for patience (in early stopping): %d' % patience)

    earlyStopping = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        verbose=2, 
        patience=patience)
    
    best_weights_filepath = os.path.join(args.model_dir, 'model_weights.hdf5')
    mcp_save = ModelCheckpoint(
        monitor='val_loss', 
        mode='min', 
        verbose=1, 
        filepath=best_weights_filepath, 
        save_best_only=True)
    
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', 
        mode='min', 
        factor=0.1, 
        patience=patience, 
        verbose=1, 
        min_delta=1e-4)

    # Parameters
    params = {'dim': (2400,),
              'batch_size': 128,
              'n_classes': 2,
              'shuffle': args.shuffle} # TODO: test effect of changing to 'True' value

    # Datasets
    ids_train, labels_train = get_partition(args.datasets_dir, 'train')
    ids_val, labels_val = get_partition(args.datasets_dir, 'validation')
    ids_test, labels_test = get_partition(args.datasets_dir, 'test')

    assert(ids_train.shape == labels_train.shape)
    assert(ids_val.shape == labels_val.shape)
    assert(ids_test.shape == labels_test.shape)

    print('#training examples:', ids_train.shape)
    print('#validation examples:', ids_val.shape)
    print('#test examples:', ids_test.shape)

    # Generators
    training_generator = DataGenerator(args.datasets_dir, 'train', ids_train, labels_train, **params)
    validation_generator = DataGenerator(args.datasets_dir, 'validation', ids_val, labels_val, **params)

    # Model fitting
    history = model.fit_generator(
                    generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=2,
                    epochs=epochs,
                    callbacks=[earlyStopping, mcp_save, reduce_lr_loss])

    plot_losses(history)

    json_filename = os.path.join(args.model_dir, "model_arch.json")
    # serialize model to JSON
    model_json = model.to_json()
    with open(json_filename, "w") as json_file:
        json_file.write(model_json)
    print("Saved model architecture to %s" % json_filename)

    # load json and create model
    json_file = open(json_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(best_weights_filepath)
    print("Loaded model weights from disk")

    np.set_printoptions(suppress=True) #prevent numpy exponential 
                                   #notation on print, default False

    # Convention used for the values in tr_y:
    #   0 --> dissimilar sentences
    #   1 --> similar sentences

    params_test = {'dim': (2400,),
              'batch_size': 1, #https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators
              'n_classes': 2,
              'shuffle': False}
    test_generator = DataGenerator(args.datasets_dir, 'test', ids_test, labels_test, **params_test)

    predicted_distances = loaded_model.predict_generator(test_generator, verbose=1)

    print(type(predicted_distances))
    print(len(predicted_distances))
    print(type(predicted_distances[0]))
    print('predicted_distances.shape:', predicted_distances.shape)

    y_true = labels_test
    y_pred = predicted_distances.argmax(axis=-1)

    print('shapes in prediction:')    
    print('ids_test.shape:', ids_test.shape)
    print('labels_test.shape: ', labels_test.shape)
    print('y_pred.shape:', y_pred.shape)

    comparativo = np.column_stack((y_true,y_pred,predicted_distances))
    print(comparativo.shape)
    print(comparativo)
    np.savetxt("foo.csv", comparativo, delimiter=",", fmt='%f')

    print('** Classification report **')
    print(classification_report(y_true, y_pred))

    print('** Confusion matrix **')
    print(confusion_matrix(y_true, y_pred))
    
'''
    Execution example:
        python train.py --datasets_dir /mnt/sdc/ebezerra/datasets100 --model_dir ../siamese_model
'''
if __name__ == "__main__":
    seed(1)
    tf.random.set_seed(2)
    main()


