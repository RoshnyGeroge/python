import numpy as np
import math
def direction(x):
 return x/np.linalg.norm(x)

def geometric_dot_product(x,y, theta):
 x_norm = np.linalg.norm(x)
 y_norm = np.linalg.norm(y)
 return x_norm * y_norm * math.cos(math.radians(theta))

def dot_product(x,y):
 result = 0
 for i in range(len(x)):
   result = result + x[i]*y[i]
 return result

x = [3,4]
print(np.linalg.norm(x)) # 5.0
print( direction(x))# [0.6 0.8]
print( direction(x))# [0.6 0.8]
print(np.linalg.norm(np.array([0.6, 0.8])))# norm of unit vectoris always 1

theta = 45
x = [3,5]
y = [8,2]
print(geometric_dot_product(x, y, theta)) #34.00000000000001

#algebraic definition of dot product
x = [3,5]
y = [8,2]
print(dot_product(x,y)) # 34

#np function
print(np.dot(x,y)) # 34
