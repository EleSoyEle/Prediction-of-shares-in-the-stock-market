import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Dense,LSTM,Input
from tensorflow.keras.models import Model
import io

#Read the file with its path
def read_file(path,flip=True,divisor=1000):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')  #Read the file
    #We take the data and scale it for that the data is divided over 1000
    #(the divisor can change depending on the minimum and maximum data)
    numbers = np.array(lines,'float32')/divisor
    #
    numbers = np.flip(numbers)
    return numbers

#Create batches of data from a data set for network training and prediction
def prepare_data(data,idx_min,idx_max,input_batch=4):
    x_data = []
    y_data = []
    data = np.array(data[idx_min:idx_max]).reshape(idx_max//(input_batch+1),input_batch+1,1)
    n = np.array(data,'float32').shape[0]
    for i in range(n):
        x_data.append(data[i][0:input_batch])
        y_data.append(data[i][input_batch])
    x_data = np.array(x_data,'float32')
    y_data = np.array(y_data,'float32')
    return x_data,y_data

#Create the model
#Input shape of the model is (None, 4,1)
def Create_model():
    input_data = Input(shape=[4,1])
    layers = [
        LSTM(64),                   #In this case an LSTM layer is used that works to understand time sequences
        Dense(64,activation="relu"),    
    ]
    last = Dense(1)
    x = input_data
    for layer in layers:
        x = layer(x)
    last = last(x)
    return Model(input_data,last)   #Output shape of the model is (None,1,1)

@tf.function
def train_step(x,y,model,optimizer,loss_object):
    with tf.GradientTape() as tape:
        prediction = model(x,training=True)
        #We calculate the loss with the predicted value and the expected value
        loss_model = loss_object(y,prediction)
        grads = tape.gradient(loss_model,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))


def train(epochs,model,train_dataset_x,train_dataset_y,optimizer,loss_object):
    for epoch in range(1,epochs+1):
        print("Epoch {}".format(epoch))
        for x,y in zip(train_dataset_x,train_dataset_y):
            train_step(x,y,model,optimizer,loss_object)


def graph_function(model,data,data_min,data_max,input_batch,multiplier=1000,save=False):
    #We copy the data
    data_2 = np.copy(data)
    for i in range(data_min,data_max):
        #We go back the index according to the batch size in this case it will be 4
        dt = np.array(data[i-input_batch:i],'float32').reshape(1,input_batch,1)
        pred = model(dt)
        data_2[i] = np.array(pred[0][0])
    #Create the plot
    plt.title("Prediction")
    plt.plot(data[data_min:data_max]*multiplier,label="Real")
    plt.plot(data_2[data_min:data_max]*multiplier,label="Predicted")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    if save:
        plt.savefig("Figure.png")
    plt.show()