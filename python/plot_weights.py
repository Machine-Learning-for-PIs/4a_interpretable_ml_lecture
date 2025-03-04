import pickle
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tikzplotlib as tikz

weights = pickle.load(open('./python/network_dump.pkl', 'rb'))


kernel_1 = weights['params']['Conv_0']['kernel']

row = kernel_1[:,:,0,0]
rows = []
for i in range(1, 32):
    if i % 6 == 0:
        rows.append(row)
        rows.append(-jnp.ones((row.shape[0], 1)))
        row = kernel_1[:,:,0,i]
    else:
        row = jnp.concatenate([row, -jnp.ones((1,3)), kernel_1[:,:,0,i]])

image = jnp.concatenate(rows, 1).T
image = jnp.concatenate((-jnp.ones((1, 23)), image), 0)
image = jnp.concatenate((-jnp.ones((21, 1)), image, -jnp.ones((21, 1))), 1)
plt.imshow(image)
plt.colorbar()
tikz.save('mnist_cnn.tex', standalone=True)
plt.show()
print('stop')