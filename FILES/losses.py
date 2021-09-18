from tensorflow.keras import backend as K 
import math
import tensorflow as tf 
from tensorflow import keras

bce = tf.keras.losses.BinaryCrossentropy()

def dice(targets, inputs, smooth=1e-6):    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)   
    intersection = K.sum(targets * inputs) 
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)  
    return dice

def dice_loss(y_true,y_pred):
   dice_loss = 1 - dice(y_true,y_pred)
   return dice_loss


def bce_logdice_loss(y_true, y_pred):
    return bce(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma) 

def DiceBCELoss(targets, inputs, smooth=1e-6):  

    y_true_f= K.flatten(targets)
    y_pred_f= K.flatten(inputs)
    BCE =  bce(y_true_f, y_pred_f)
    intersection = K.sum(y_true_f * y_pred_f)

    dice_loss = 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    Dice_BCE = BCE + dice_loss 

   
    return Dice_BCE

def log_cosh_dice_loss(y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

def IoU(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(targets * inputs)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return IoU