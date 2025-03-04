import pickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tikzplotlib as tikz
from mnist import CNN


weights = pickle.load(open('./python/network_dump.pkl', 'rb'))
net = CNN()
neuron = 6

def forward_pass(x):
    out = net.apply(weights, x)
    return out[0, neuron]

get_grads = jax.value_and_grad(forward_pass)

x = jax.random.uniform(jax.random.PRNGKey(42),
        [1, 28, 28, 1])
x = jnp.ones([1, 28, 28, 1])

grads = []
for i in range(25):
    x = (x - jnp.mean(x))/jnp.std(x + 1e-5)
    act, grad = get_grads(x)
    x = x + grad
    grads.append(grad)
    print(act)

mean_grad = jnp.mean(jnp.stack(grads, 0), 0)

plt.imshow(x[0, :, :, 0])
plt.title('Input maxizing the ' + str(neuron) + '- neuron')
# tikz.save('grad_plot_mnist_6.tex', standalone=True)
plt.show()
