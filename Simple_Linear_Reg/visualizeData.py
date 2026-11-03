import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import fb, fw

nbr_simples = 1000
data = pd.read_csv('Simple_Linear_Reg/datasets/train_dataset.csv', nrows=nbr_simples)

studyHours_X = data['Study_Hours'].to_numpy()
cgpa_Y = data['CGPA'].to_numpy()

y_pred_line = fw * studyHours_X + fb

plt.figure(figsize=(9, 6))

plt.scatter(
    studyHours_X, 
    cgpa_Y, 
    color='#2b5c8f',
    s=35,
    alpha=0.5,        
    edgecolors='white',    
    linewidth=0.5
)

plt.plot(
    studyHours_X, 
    y_pred_line, 
    color='red', 
    linewidth=2, 
    label=f'Line: y = {fw:.2f}x + {fb:.2f}'
)

plt.title('Linear Regression Fit', fontsize=14, fontweight='bold')
plt.xlabel('Study Hours', fontsize=11)
plt.ylabel('CGPA', fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)


plt.show()
