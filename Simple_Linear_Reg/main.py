import numpy as np
import pandas as pd
from train import train_linear_reg_model
alpha = 0.001;
w = b = 0.0
epochs = 15000

nbr_simples = 1000
data = pd.read_csv('Simple_Linear_Reg/datasets/train_dataset.csv', nrows=nbr_simples)

studyHours_X = data['Study_Hours'].to_numpy()
cgpa_Y = data['CGPA'].to_numpy()


training_result = train_linear_reg_model(epochs, nbr_simples, studyHours_X, cgpa_Y, alpha, w, b)
fw = training_result[0]
fb = training_result[1]
print("Final Slop : ", fw)
print("Final Bias : ", fb)



