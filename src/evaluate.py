from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
import matplotlib.pyplot as plt

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def evaluate(individual):

    print('\n-----------genome evaluation-------------')
    
    num_params = len(func_seq)
    #n = len(genome)
    input_layer_flag = False
    return_flag = False

    model = Sequential() 

    for i in range(IND_SIZE):
        
        index = i*num_params
        
        if individual[index] > 0:
            if input_layer_flag==False:
                model.add(LSTM(individual[index+1],activation=individual[index+2],
                            input_shape=(lags, features),
                            kernel_regularizer= keras.regularizers.l2(0.0001),
                                return_sequences=True))
        
                input_layer_flag=True
        
            else:
                model.add(LSTM(individual[index+1],activation=individual[index+2],
                            kernel_regularizer= keras.regularizers.l2(0.0001), 
                            return_sequences=True))
        
            return_flag=True
    
    # final layer
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    model.fit(X_train, y_train, batch_size = 2, validation_data= (X_val, y_val) ,  epochs=2 ,verbose=1)
    y_pred = model.predict(X_val)
    
    return mean_squared_error(y_val, y_pred),