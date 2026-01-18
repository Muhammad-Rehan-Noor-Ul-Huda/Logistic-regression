import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g
def cost(x,y,w,b):
    m=len(x)
    cost_f=0
    for i in range(m):
        z=w*x[i]+b
        z_g=sigmoid(z)
        cost_f+=y[i]*math.log(z_g)+(1-y[i])*math.log(1-z_g)
    cost_f/=-m
    return cost_f
def gradient(x,y,w,b):
    dw=0
    db=0
    m=len(x)
    for i in range(m):
        z=w*x[i]+b
        y_h=sigmoid(z)
        dw+=(y_h-y[i])*x[i]
        db+=(y_h-y[i])
    dw/=m
    db/=m
    return dw,db
def final(x,y,w,b,iter,alpha):
    for i in range(iter):
        dw,db=gradient(x,y,w,b)
        w=w-alpha*dw
        b=b-alpha*db
        if i%100==0:
            cost_c=cost(x,y,w,b)
            print(f"iterations : {i}  w  : {w}  b :  {b}  cost :  {cost_c}")
    return w,b
data = pd.read_csv('discount_purchase_data.csv')
x=data.iloc[:,0]
y=data.iloc[:,1]
plt.scatter(x,y)
b=0
w=0
iter=3500
alpha=0.0201
w_f,b_f=final(x,y,w,b,iter,alpha)
z_final=w_f*x+b_f
y_final=sigmoid(z_final)
x_curve = np.linspace(min(x), max(x), 200)   # 200 evenly spaced points
y_curve = sigmoid(w_f * x_curve + b_f)
plt.plot(x_curve, y_curve, color="red", label="Sigmoid Curve")

plt.xlabel('discount', color='black')
plt.ylabel('purchase decision')
plt.title('Discount purchase model!')
plt.legend()
plt.show()
check=int(input("Enter discount to check it would be sold or not! "))
z_check=w_f*check+b_f
y_check=sigmoid(z_check)
if y_check>=0.5:
    y_told=1
    print("Yes it Would be Sold!")
else:
    y_told=0
    print("No, it Would not be Sold!")
0.000253


