from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

w = 6
x=np.linspace(-w,w, num=20)
y=np.linspace(-w,w, num=20)

x, y = np.meshgrid(x, y)

z = np.exp(-0.1*x**2-0.1*y**2) 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,z, alpha=0.2,linewidth=0.5, edgecolors='black')
ax.axis('off')
#plt.show()
plt.tight_layout()
plt.savefig('gaussian.png')