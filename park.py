from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import pickle

#Archivo de texto 
DATA_DIR = "./data2.txt" 
#Modificar BATCH_SIZE o HIDDEN_DIM en caso tengan problemas de memoria
BATCH_SIZE = 60
HIDDEN_DIM = 130 #500
#Parametro para longitud de secuencia a analizar
SEQ_LENGTH = 100
#Parametro para cargar un pesos previamente entrenados (checkpoint)
WEIGHTS = '' 

#Parametro para indicar cuantos caracteres generar en cada prueba
GENERATE_LENGTH = 300 
#Parametros para la red neuronal
LAYER_NUM = 2 
NB_EPOCH = 20

# method for preparing the training data
def load_data(data_dir, seq_length):
    #Carga del archivo
    data = open(data_dir, 'r').read()
    #Caracteres unicos
    chars = list(set(data))
    VOCAB_SIZE = len(chars)

    print('Data length: {} characters'.format(len(data)))
    print('Vocabulary size: {} characters'.format(VOCAB_SIZE))
    print(chars)
    
    #Indexacion de los caracteres
    ix_to_char = {ix:char for ix, char in enumerate(chars)}
    char_to_ix = {char:ix for ix, char in enumerate(chars)}
    
    #Estructuras de entrada y salida
    NUMBER_OF_SEQ = int(len(data)/seq_length)
    print('Number of sequences: {}'.format(NUMBER_OF_SEQ))
    X = np.zeros((NUMBER_OF_SEQ, seq_length, VOCAB_SIZE))
    y = np.zeros((NUMBER_OF_SEQ, seq_length, VOCAB_SIZE))
    
    for i in range(0, NUMBER_OF_SEQ):
        #LLenado de la estructura de entrada X
        X_sequence = data[i*seq_length:(i+1)*seq_length]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]
        #one-hot-vector (input)
        input_sequence = np.zeros((seq_length, VOCAB_SIZE))  
        #uso del diccionario para completar el one-hot-vector
        for j in range(seq_length):
            input_sequence[j][X_sequence_ix[j]] = 1.
            X[i] = input_sequence
            
        #Llenado de la estructura de salida y
        y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        #one-hot-vector (output)
        target_sequence = np.zeros((seq_length, VOCAB_SIZE))
        #uso del diccionario para completar el one-hot-vector
        for j in range(seq_length):
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[i] = target_sequence
            
    return X, y, VOCAB_SIZE, ix_to_char

# method for generating text
def generate_text(model, length, vocab_size, ix_to_char):
    # starting with random character
    ix = [np.random.randint(vocab_size)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, vocab_size))
    for i in range(length):
        # appending the last predicted character to sequence
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)

# Creating training data
X, y, VOCAB_SIZE, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)

# Creating and compiling the Network
model = Sequential()

#Añadiendo las capas LSTM
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
#Añadiendo la operacion de salida
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))

#"Compilando" = instanciando la RNN con su función de pérdida y optimización
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

#Cuidar de no reemplazar el pickle original
with open('ix_to_char.pickle', 'rb') as handle:
    ix_to_char = pickle.load(handle)

WEIGHTS ="checkpoint_layer_2_hidden_300_epoch_20.hdf5"
# Loading the trained weights
model.load_weights(WEIGHTS)
generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
print('\n\n')