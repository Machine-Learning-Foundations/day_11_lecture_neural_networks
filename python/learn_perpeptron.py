import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt


def relu(x):
    x = x.at[x < 0].set(0)
    return x


def sigmoid(x):
    return 1./(1. + jnp.exp(-x))


def net(params):
    h0 = sigmoid(params['W']@params['x'] + params['b'])
    h1 = params['W1']@h0 # + params['b1']
    return h1


def cost(y, h):
    return jnp.sum((y - h)**2)

def net_cost(params):
    h = net(params)
    return cost(params['y'], h)

if __name__ == '__main__':

    step_size = .005
    iterations = 100

    hidden_neurons = 10

    dense_key = jax.random.PRNGKey(0)
    dense_key, bias_key = jax.random.split(dense_key)
    bias_key, noise_key = jax.random.split(bias_key)
    
    noise_key, dense_key2 = jax.random.split(noise_key)
    dense_key2, bias_key2 = jax.random.split(dense_key2)
    
    
    W = jax.random.uniform(dense_key, [hidden_neurons, 200], minval=-1, maxval=1.)
    b = jax.random.uniform(bias_key, [hidden_neurons], minval=-1, maxval=1.)

    W1 = jax.random.uniform(dense_key2, [200, hidden_neurons], minval=-1, maxval=1.)

    x = jnp.linspace(-3*jnp.pi, 3*jnp.pi, 200)
    y = jnp.cos(x) 
    
    for i in range(iterations):
        y_noise = y + jax.random.normal(noise_key, [200])

        value_grads = jax.value_and_grad(net_cost)
        current_cost, grads = value_grads({"x": y_noise, "y": y, "W": W, "b": b,
                                          "W1": W1})

        W += - step_size * grads["W"]
        b += - step_size * grads["b"]
        W1 += - step_size * grads["W1"]

        print(i, current_cost)

    y_hat = net({"x": y_noise, "y": y, "W": W, "b": b,
                "W1": W1})

    plt.plot(x, y)
    plt.plot(x, y_hat, '.')
    plt.show()

    print('done')