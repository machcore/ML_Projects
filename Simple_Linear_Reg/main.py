import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import train_linear_reg_model
import matplotlib.ticker as ticker



alpha = 0.0001
w = b = 0.0
epochs = 100000
nbr_simples = 1000

data = pd.read_csv('Simple_Linear_Reg/datasets/train_dataset.csv', nrows=nbr_simples)
studyHours_X = data['Study_Hours'].to_numpy()
cgpa_Y = data['CGPA'].to_numpy()




# Train model and receive loss_history
w, b, loss_history = train_linear_reg_model(epochs, nbr_simples, studyHours_X, cgpa_Y, alpha, w, b)


test_data = pd.read_csv('Simple_Linear_Reg/datasets/test_dataset.csv')
test_data["pred_cgpa"] = w * test_data["Study_Hours"] + b
test_data.to_csv("Simple_Linear_Reg/datasets/test_with_pred.csv", index=False)

# Plot Loss vs Epochs

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, color='#e74c3c', linewidth=2)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.title('Loss Variation per Epoch', fontsize=13, fontweight='bold')
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss (MSE)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()