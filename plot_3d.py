import numpy as np
import matplotlib.pyplot as plt


C = 100
Cy = C
Cx = 2 * C + 1
ME = 2
NORM = (1 + ME * ME) ** 0.5

a1 = np.array([[0.0] * Cy for _ in range(Cx)])
a2 = np.array([[0.0] * Cy for _ in range(Cx)])
a1[C][0] = 0.0
a2[C][0] = 1.0

for y in range(1, Cy):
    for x in range(-y + C, y + C + 1, 2):
        a1[x][y] = (a2[x + 1][y - 1] * ME + a1[x + 1][y - 1]) / NORM
        a2[x][y] = (a2[x - 1][y - 1] - a1[x - 1][y - 1] * ME) / NORM

p = (a1 ** 2 + a2 ** 2)

val = -1
anglex = -1
for x in range(Cx):
    if p[x][-1] > val:
        val = p[x][-1]
        anglex = x

anglex -= C
angley = Cy
# print(anglex, angley)
# print(anglex / angley)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(-C, C + 1, 1)
y = np.arange(0, Cy, 1)
X, Y = np.meshgrid(x, y, indexing="ij")
zs = p
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z, alpha=0.6, rcount=50, ccount=50)
ax.set_xlabel('X Label 1')
ax.set_ylabel('Y Label 2')
ax.set_zlabel('Z Label 3')
plt.show()
