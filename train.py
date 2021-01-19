import tensorflow as tf
import numpy as np
from utils import *
import os
import sys

path = "datos_de_cierre.txt"
numbers = read_file(path,flip=True,divisor=1000)

#The input_batch is the number of data that we will give to the model so that based on them it can make the prediction
input_batch = 4

#Training data range
data_min = 0
data_max = 700

#We prepare the data to be able to train the model
data_x,data_y = prepare_data(numbers,data_min,data_max,input_batch=input_batch)

#We create the dataset
train_dataset_x = tf.data.Dataset.from_tensor_slices(data_x).batch(1)
train_dataset_y = tf.data.Dataset.from_tensor_slices(data_y).batch(1)

#We create the model
predictor = Create_model()

#We define the lr
#In this case, 1x10⁻³ is enough to train
lr = 1e-3

#We are going to use the Adam optimizer
optimizer = tf.keras.optimizers.Adam(lr,beta_1=0.5)

#The error function can be another but in this case consider that it would be the best
loss_object = tf.keras.losses.MeanSquaredError()

checkpoint_path = "./checkpoints/"
checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")

checkpoint = tf.train.Checkpoint(
    predictor=predictor,
    optmizer=optimizer
)

#We check if there are checkpoints and if there are, it will ask whether to restore or not
if not os.listdir(checkpoint_path) == []:
    print("There are already previous checkpoints")
    rest = str(input("You want to restore the data of your model?[y/n]: "))
    if rest.lower() == "y":
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
        print("Data restored")


print("Your model will be trained")

#The number of steps must vary in order to predict
epochs = 1000
train(epochs,predictor,train_dataset_x,train_dataset_y,optimizer,loss_object)

#This saves the data for your use
checkpoint.save(checkpoint_prefix)

#We plot the data in a certain range
val_min = 700
val_max = 764

graph_function(predictor,numbers,val_min,val_max,input_batch,multiplier=1000)