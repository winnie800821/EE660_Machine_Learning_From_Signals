import numpy as np
import matplotlib.pyplot as plt
import random

x = np.linspace(-1,1,1000)
target_y=x**2
array_a=np.zeros(1000)
array_b=np.zeros(1000)
for i in range(1000):
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    a = x1 + x2
    b = -x1 * x2
    array_a[i]=a
    array_b[i]=b
    y_head = a * x + b
    Eout=(y_head-x**2)**2
    plt.plot(x, y_head, color="red")
plt.plot(x,target_y, color="blue",label='target function')
avg_ga=np.mean(array_a)
avg_gb=np.mean(array_b)
plt.plot(x,avg_ga*x+avg_gb, color="green",label='average_g')
plt.legend()
plt.xlim(-1, 1)
plt.ylim(-2, 5)
plt.show()

array_Eout=np.zeros(1000)
array_var=np.zeros(1000)
array_bias=np.zeros(1000)
array_Eout = (array_a*x+array_b-x**2)**2
Eout=np.mean(array_Eout)
array_var=((array_a*x+array_b)-(avg_ga*x+avg_gb))**2
var=np.mean(array_var)
array_bias=((avg_ga*x+avg_gb)-x**2)**2
bias=np.mean(array_bias)
print('g_head(x)=%.2f *x + %.2f' %(avg_ga,avg_gb))
print("Eout=",Eout)
print("Var=",var)
print("bias=",bias)
print('Var+bias=',var+bias)


