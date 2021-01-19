import tensorflow as tf
import numpy as np
from utils import *
import os
import sys


path = "datos_de_cierre.txt"
numbers = read_file(path,flip=True,divisor=1000)

#The input_batch is the number of data that we will give to the model so that based on them it can make the prediction
input_batch = 4

predictor = Create_model()

checkpoint_path = "./checkpoints/"
checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")

checkpoint = tf.train.Checkpoint(
    predictor=predictor
)

#We restore the model data
print("Restoring data")
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
print("Data restored")

#We plot the data in a certain range
val_min = 700
val_max = 764

graph_function(predictor,numbers,val_min,val_max,input_batch,multiplier=1000)