import numpy as np
import pandas as pd
import oper as op

def train_linear_reg_model(epochs, n, x, y, alpha, w, b):
    loss_history = []  # 1. Initialize empty list BEFORE the loop

    for epoch in range(epochs):
        predicted_cgpa = op.cal_Pred_vals(w, b, x)
        loss = op.lossFunc(n, predicted_cgpa, y)
        loss_history.append(loss)  # 2. Append loss on EVERY epoch
        
        grd_w = op.gradient_w(predicted_cgpa, y, x, n)
        grd_b = op.gradient_b(predicted_cgpa, y, n)
        
        w = op.new_w(w, grd_w, alpha)
        b = op.new_b(b, grd_b, alpha)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:4d} | Loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")

    print("\nFinal Parameters:")
    print(f"Optimal Slope (w): {w:.4f}")
    print(f"Optimal Bias  (b): {b:.4f}")

    return w, b, loss_history