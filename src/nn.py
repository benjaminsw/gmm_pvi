from collections.abc import Callable
import jax
import jax.numpy as np
import jax.nn as nn
import equinox as eqx
import jax.tree_util as jtu


class Module(eqx.Module):
    def get_filter_spec(self):
        filter_spec = jtu.tree_map(eqx.is_array,
                                   self)
        return filter_spec


class Net(Module):
    d_in : int
    d_out : int
    fc1 : eqx.nn.Linear
    fc2 : eqx.nn.Linear
    fc3 : eqx.nn.Linear
    act : Callable

    def __init__(self,
                 key: jax.random.PRNGKey,
                 d_in: int,
                 d_out: int,
                 n_hidden: int,
                 act: Callable):
        fc1_key, fc2_key, fc3_key = jax.random.split(key, 3)
        self.fc1 = eqx.nn.Linear(d_in, n_hidden, key=fc1_key)
        self.fc2 = eqx.nn.Linear(n_hidden, n_hidden, key=fc2_key)
        self.fc3 = eqx.nn.Linear(n_hidden, d_out, key=fc3_key)
        self.d_in = d_in
        self.d_out = d_out
        self.act = act

    def __call__(self, x: jax.Array):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_filter_spec(self):
        filter_spec = jtu.tree_map(eqx.is_array,
                                   self)
        return filter_spec

class XYNet(Module):
    net: eqx.Module
    d_x: int
    d_y : int
    def __init__(self,
                 key: jax.random.PRNGKey,
                 d_x: int,
                 d_z: int,
                 d_y: int,
                 n_hidden: int):
        self.net = Net(key,
                    d_z + d_y,
                    d_x * 2,
                    n_hidden,
                    act=jax.nn.leaky_relu)
        self.d_x = d_x
        self.d_y = d_y

    def __call__(self, z: jax.Array, y: jax.Array):
        x = self._format_input(z, y)
        x = self.net(x)
        mu = x[:self.d_x]
        var = nn.softplus(x[self.d_x:]) + 1e-8
        return mu, var
        
    def _format_input(self, z: jax.Array, y: jax.Array = None):
        assert z.ndim == 1, f"Shape of z is {z.shape}"
        if not self.d_y == 0:
            assert y.ndim == 1, f"Shape of y is {y.shape}"
            return np.concatenate([z, y], axis=-1)
        else:
            return z

    def get_filter_spec(self):
        filter_spec = jtu.tree_map(eqx.is_array,
                                   self)
        filter_spec = eqx.tree_at(lambda tree : tree.net,
                                  filter_spec,
                                  self.net.get_filter_spec())
        return filter_spec