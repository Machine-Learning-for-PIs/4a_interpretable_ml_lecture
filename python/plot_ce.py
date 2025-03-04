import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tikz

def scalar_ce(o,y):
    return (-1)*(y*np.log(o)+(1 - y)*np.log(1-o))

o = np.linspace(.01, .99, 200)
y = np.linspace(.01, .99, 200)
mo,my = np.meshgrid(o,y)
plt.imshow(scalar_ce(mo,my), extent=[.01, .99,.99,.01], origin='upper')
plt.xlabel('o')
plt.ylabel('y')
plt.colorbar()
ax = plt.gca()
ax.invert_yaxis()
plt.show()


plt.plot(o, scalar_ce(o, 1.));
plt.title('Cross entropy for label equal 1')
plt.ylabel('cost')
plt.xlabel('network output')
tikz.save('ce_label_1.tex', standalone=True)
plt.show()

plt.plot(o, scalar_ce(o, 0));
plt.title('Cross entropy for label equal 0')
plt.ylabel('cost')
plt.xlabel('network output')
tikz.save('ce_label_0.tex', standalone=True)
plt.show()