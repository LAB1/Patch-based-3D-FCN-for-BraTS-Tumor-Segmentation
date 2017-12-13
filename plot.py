import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import argparse
import sys
import tempfile
import os
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
import tempfile
import math
from tensorflow.examples.tutorials.mnist import input_data

import numpy.ma as ma


acc_list = np.load("./acc_list_norm.npy")
dice_list = np.load("./dice_list_norm.npy")
no_dice_list = np.load("./no_dice_list.npy")
banchmark_2015 = np.full((len(dice_list)),0.87)
state_of_art = np.full((len(dice_list)),0.92)
mx = ma.masked_array(dice_list, mask = (dice_list < 0.3) )
x = np.arange(len(no_dice_list))
plt.xlabel("iter")
""""
plt.plot(x, acc_list, label = "accuracy")
plt.plot(x, dice_list, label="dice score")
plt.plot(x, banchmark_2015, label="Pereira ‎2015")
plt.plot(x, state_of_art, label="Erden 2017")
"""
plt.plot(x, dice_list[:len(no_dice_list)], label="BN dice score")
plt.plot(x, no_dice_list, label="non-BN dice score")

plt.legend(loc = 1, borderaxespad = 0)
plt.show()