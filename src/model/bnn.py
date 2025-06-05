import jax
import jax.numpy as np
from jax.scipy.stats import norm

class BNN:
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 n_hidden: int,
                 transform=None,
                 scale: float=1.):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hidden = n_hidden
        self.layer1_dim = (in_dim + 1) * n_hidden
        self.layer2_dim = (n_hidden + 1) * out_dim
        self.dim = self.layer1_dim + self.layer2_dim
        self.transform = transform if transform is not None else lambda x: x
        self.scale = scale
        self.prior_scale = 5.
    
    def get_w_b(self, x):
        w1 = x[:self.in_dim * self.n_hidden].reshape(self.n_hidden, self.in_dim)
        b1 = x[self.in_dim * self.n_hidden:self.layer1_dim].reshape(self.n_hidden)
        w2 = x[self.layer1_dim:self.layer1_dim + self.n_hidden * self.out_dim].reshape(self.out_dim, self.n_hidden)
        b2 = x[self.layer1_dim + self.n_hidden * self.out_dim:].reshape(self.out_dim)
        return w1, b1, w2, b2

    def eval_f(self, x, loc):
        w1, b1, w2, b2 = self.get_w_b(x)
        acc = np.einsum("ij,...j->...i", w1, loc) + b1
        acc = jax.nn.relu(acc)
        acc = np.einsum("ij,...j->...i", w2, acc) + b2
        acc = self.transform(acc)
        return acc

    def log_likeli(self, x, y):
        out, loc = y
        m_out = self.eval_f(x, loc)
        return norm.logpdf(out,
                           loc=m_out,
                           scale=self.scale).sum(-1)
    
    def log_prior(self, x):
        return norm.logpdf(x,
                           scale=self.prior_scale).sum(-1)