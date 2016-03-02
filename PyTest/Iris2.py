from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
features = data['data']
target = data['target']
is_setosa = (target == 0)
is_versicolor = (target ==1)
is_virginica = (target == 2)
setosa_feature = features[is_setosa]
versicolor_feature = features[is_versicolor]
virgincia_feature = features[is_virginica]


x1 = setosa_feature[:,0]
y1 = setosa_feature[:,1]
x2 = versicolor_feature[:,0]
y2 = versicolor_feature[:,2]
z1 = virgincia_feature[:,0]
z2 = virgincia_feature[:,2]
plt.plot(y1,z1,'o')
plt.plot(y2,z2,'x')
plt.show()

x_ratio = abs(x1.mean() -x2.mean())/max(x1.mean(),x2.mean())
y_ratio = abs(y1.mean()-y2.mean())/max(y1.mean(),y2.mean())
z_ratio = abs(z1.mean()-z2.mean())/max(z1.mean(),z2.mean())
total_ratio_xy = max(x_ratio,y_ratio)/min(x_ratio,y_ratio)
total_ratio_yz = max(y_ratio,z_ratio)/min(y_ratio,z_ratio)
print y1
print y2
y2 = total_ratio_yz*y2
y1 = y1/total_ratio_yz
z1 = z1
z2 = z2
plt.plot(y1,z1,'o')
plt.plot(y2,z2,'x')
plt.show()