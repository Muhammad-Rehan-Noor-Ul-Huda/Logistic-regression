import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(x,y,w,b):
    m=len(x)
    final_cost=0
    for i in range(m):
        z=np.dot(w,x.iloc[i,:])+b
        y_hat=sigmoid(z)
        final_cost+=y[i]*math.log(y_hat)+(1-y[i])*math.log(1-y_hat)
    final_cost/=-m
    return final_cost

def gradient(x,y,w,b):
    dw=np.zeros(len(w))
    db=0
    m=len(x)
    for i in range(m):
        z=np.dot(w,x.iloc[i,:])+b
        y_hat=sigmoid(z)
        dw+=(y_hat-y[i])*x.iloc[i, :]
        db+=(y_hat-y[i])
    dw/=m
    db/=m
    return dw,db

def final(x,y,w,b,alpha,iter):
    for i in range(iter):
        dw,db=gradient(x,y,w,b)
        w=w-alpha*dw
        b=b-alpha*db
        # if i % 100 == 0:
        print(f"cost function at iteration {i} = {cost(x,y,w,b)}")
    return w,b
data=pd.read_csv('diabetes.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
b=0
m,n=x.shape
w=np.zeros(n)


iter=5000
alpha=0.000245

if os.path.exists("logistic_model.npz"):
    print("Loading saved model...")
    model = np.load("logistic_model.npz")
    w_final = model["w"]
    b_final = model["b"]
else:
    print("Training new model...")
    w_final,b_final=final(x,y,w,b,alpha,iter)
    np.savez("logistic_model.npz", w=w_final, b=b_final)
    print("Model saved to logistic_model.npz")


fixed_features = x.mean().values  
x_curve = np.linspace(x['Glucose'].min(), x['Glucose'].max(), 200)
y_curve = []

for val in x_curve:
    features = fixed_features.copy()
    features[0] = val  
    z = np.dot(w_final, features) + b_final
    y_curve.append(sigmoid(z))

plt.scatter(x['Glucose'], y, color='blue', label='Actual Data')
plt.plot(x_curve, y_curve, color='red', label='Predicted Probability')
plt.xlabel('Glucose')
plt.ylabel('Probability of Diabetes')
plt.title('Glucose vs Diabetes Prediction')
plt.legend()
plt.show()
# Predict on training data
z = np.dot(x, w_final) + b_final
y_pred_prob = sigmoid(z)
y_pred = (y_pred_prob >= 0.5).astype(int)

accuracy = np.mean(y_pred == y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")
