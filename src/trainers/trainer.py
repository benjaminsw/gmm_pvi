import sys
import jax
from tqdm import tqdm
import equinox as eqx
from jaxtyping import PRNGKeyArray as PRNGKey
from src.base import Target
from jaxtyping import Float, Array
from typing import Callable, Dict
import jax.numpy as np
from collections import defaultdict
from math import ceil
import time


def trainer(key: PRNGKey,
            carry: Dict,
            target: Target,
            ys: Float[Array, "..."],
            step: Callable,
            max_epochs: int,
            metrics=None,
            use_jit=True):
    '''
    step : Callable[[PRNGKey, Dict, Target, Array], Tuple[float, Dict]] 
    '''
    if use_jit:
        step = eqx.filter_jit(step)
    history = defaultdict(list)
    pbar = tqdm(range(max_epochs), disable= not sys.stdout.isatty())
    total_time = 0
    for i in pbar:
        start_time = time.time()
        key1, key = jax.random.split(key, 2)
        lval, carry = step(key1,
                           carry,
                           target,
                           ys)
        pbar.set_description(f"Loss {lval:.3f}")
        end_time = time.time()
        total_time += end_time - start_time
        assert np.isnan(lval) == False, "Loss is NaN"
        if i % 100 == 0:
            history['loss'].append(lval)
            history['time'].append(total_time)
            if metrics is not None:
                metrics_key, key = jax.random.split(key, 2)
                metric_dict = metrics(metrics_key,
                                    carry.id,
                                    target)
                for k, v in metric_dict.items():
                    history[k].append(v)
    return history, carry


def subsample_trainer(
        key: PRNGKey,
        carry: Dict,
        target: Target,
        ys: Float[Array, "..."],
        step: Callable,
        max_epochs: int,
        subsample: int,
        metrics=None,
        checkpoint=None,
        use_jit=False):
    '''
    step : Callable[[PRNGKey, Dict, Target, Array], Tuple[float, Dict]] 
    '''
    if use_jit:
        step = eqx.filter_jit(step)
    history = defaultdict(list)
    if subsample > target.train_size:
        print(f"Subsample size {subsample} changed to train size {target.train_size}")
        subsample = target.train_size
    pbar = tqdm(range(max_epochs),
                leave=False)
    for _ in pbar:
        a_lval = 0
        n_iter = ceil(target.train_size / subsample)
        index_key, key = jax.random.split(key, 2)
        indices = jax.random.choice(index_key,
                                    target.train_size,
                                    (target.train_size,),
                                    replace=False)
        for i in range(n_iter):
            step_key, key = jax.random.split(key, 2)
            y_indices = indices[i * subsample:(i + 1) * subsample]
            lval, carry = step(step_key,
                               carry,
                               target,
                               y_indices)
            assert np.isnan(lval) == False, "Loss is NaN"
            a_lval += (lval - a_lval) / (i + 1)
            pbar.set_description(f"Avg Loss {a_lval:.3f}")
        history['loss'].append(a_lval)
        if metrics is not None:
            metrics_key, key = jax.random.split(key, 2)
            metric_dict = metrics(metrics_key,
                                  carry,
                                  target)
            for k, v in metric_dict.items():
                history[k].append(v)
    for k, v in history.items():
        history[k] = np.array(v)
    return history, carry