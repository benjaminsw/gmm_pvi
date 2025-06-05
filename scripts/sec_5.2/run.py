from tqdm import tqdm
import typer
from src.problems.toy import *
from src.id import *
from src.trainers.trainer import trainer
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from pathlib import Path
from src.base import *
from src.utils import (make_step_and_carry,
                       config_to_parameters,
                       parse_config)
import pickle
from ot.sliced import sliced_wasserstein_distance
import numpy
from mmdfuse import mmdfuse


app = typer.Typer()

PROBLEMS = {
    'banana': Banana,
    'multimodal': Multimodal,
    'xshape': XShape,
}

ALGORITHMS = ['gmm_pvi','pvi'] #, 'sm', 'svi', 'uvi']

def visualize(key, 
              ids,
              target,
              path,
              prefix=""):
    _max = 4.5
    _min = -4.5
    x_lin = np.linspace(_min, _max, 1000)
    XX, YY = np.meshgrid(x_lin, x_lin)
    XY = np.stack([XX, YY], axis=-1)
    log_p = lambda x : target.log_prob(x, None)
    log_true_ZZ = vmap(vmap(log_p))(XY)
    plt.clf()
    if 'pvi' in ids:
        model_log_p = lambda x: ids['pvi'].log_prob(x, None)
        log_model_ZZ = vmap(vmap(model_log_p))(XY)

        diff = np.abs(np.exp(log_true_ZZ) - np.exp(log_model_ZZ))
        plt.imshow(diff, cmap=mpl.colormaps['Reds'])
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15) 
        cbar.ax.locator_params(nbins=5)

        c_true = plt.contour(
            np.exp(log_true_ZZ),
            levels=5,
            colors='black',
            linewidths=6,
            #linestyles='dashed',
            label='True')
        c_model = plt.contour(
            np.exp(log_model_ZZ),
            levels=c_true._levels,
            colors='deepskyblue',
            linewidths=2,
            label='Model')
        labels = ['True', 'Model']
        # Fix for matplotlib compatibility
        try:
            c_true_color = c_true.collections[-1].get_edgecolor() if hasattr(c_true, 'collections') else 'black'
            c_model_color = c_model.collections[-1].get_edgecolor() if hasattr(c_model, 'collections') else 'deepskyblue'
        except (AttributeError, IndexError):
            c_true_color = 'black'
            c_model_color = 'deepskyblue'
            
        lines = [plt.Line2D([0], [0], color=c_true_color, lw=2),
                plt.Line2D([0], [0], color=c_model_color, lw=2)]
        #plt.legend(lines, labels)
        plt.xticks([])
        plt.yticks([])

        (path / 'pvi').mkdir(exist_ok=True, parents=True)
        plt.savefig(path / 'pvi' / f"{prefix}_pdf.pdf")

    for alg, id in ids.items():
        plt.clf()
        m_key, t_key, key = jax.random.split(key, 3)
        model_samples = id.sample(m_key, 100, None)
        target_samples = target.sample(t_key, 100, None)
        c = plt.contour(XX,
                        YY,
                        np.exp(log_true_ZZ),
                        levels=5,
                        cmap='Reds',)
        plt.scatter(model_samples[..., 0],
                    model_samples[..., 1],
                    alpha=0.5,
                    label='Samples')
        plt.scatter(target_samples[..., 0],
                    target_samples[..., 1],
                    alpha=0.5,
                    label='Samples')
        plt.savefig(path / f'{alg}' / f"{prefix}_samples.pdf")

        plt.clf()
        model_samples = id.sample(key, 10000, None)
        plt.hist2d(model_samples[:, 0],
                   model_samples[:, 1],
                   bins=100,
                   cmap='Blues',
                   label='Samples')
        plt.savefig(path / f'{alg}' / f"{prefix}_ecdf.pdf")
        del model_samples
    # plt.legend()


def test(key, x, y):
    output = mmdfuse(x, y, key)
    return output


def compute_power(
        key,
        target,
        id,
        n_samples=500,
        n_retries=100):
    avg_rej = 0
    for _ in range(n_retries):
        m_key, t_key, test_key, key = jax.random.split(key, 4)
        model_samples = id.sample(m_key, n_samples, None)
        target_samples = target.sample(t_key, n_samples, None)
        avg_rej = avg_rej + test(
            test_key, model_samples, target_samples,
        )
    return avg_rej / n_retries


def compute_w1(key,
               target,
               id,
               n_samples=10000,
               n_retries=1):
    distance = 0
    for _ in range(n_retries):
        m_key, t_key, key = jax.random.split(key, 3)
        model_samples = id.sample(m_key, n_samples, None)
        target_samples = target.sample(t_key, n_samples, None)
        distance = distance + sliced_wasserstein_distance(
            numpy.array(model_samples), numpy.array(target_samples),
            n_projections=100,
        )
    return distance / n_retries


def metrics_fn(key,
            target,
            id):
    power = compute_power(
        key, target, id, n_samples=1000, n_retries=100
    ) 
    sliced_w = compute_w1(key,
                          target,
                          id,
                          n_samples=10000,
                          n_retries=10)
    return {'power': power,
            'sliced_w': sliced_w}


@app.command()
def run(config_name: str,
        seed: int=2):
    config_path = Path(f"scripts/sec_5.2/config/{config_name}.yaml")
    assert config_path.exists()
    config = parse_config(config_path)

    n_rerun = config['experiment']['n_reruns']
    n_updates = config['experiment']['n_updates']
    name = config['experiment']['name']
    name = 'default' if len(name) == 0 else name
    compute_metrics = config['experiment']['compute_metrics']
    use_jit = config['experiment']['use_jit']

    parent_path = Path(f"output/sec_5.2/{name}")
    key = jax.random.PRNGKey(seed)
    histories = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    results = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))

    for prob_name, problem in PROBLEMS.items():
        for i in tqdm(range(n_rerun)):
            trainer_key, init_key, key = jax.random.split(key, 3)
            ids = {}
            target = problem()
            path = parent_path / f"{prob_name}"
            path.mkdir(parents=True, exist_ok=True)

            for algo in ALGORITHMS:
                m_key, key = jax.random.split(key, 2)
                parameters = config_to_parameters(config, algo)
                step, carry = make_step_and_carry(
                    init_key,
                    parameters,
                    target)
                
                metrics = compute_w1 if compute_metrics else None
                history, carry = trainer(
                    trainer_key,
                    carry,
                    target,
                    None,
                    step,
                    n_updates,
                    metrics=metrics,
                    use_jit=use_jit,
                )
                ids[algo] = carry.id
                for k, v in history.items():
                    plt.clf()
                    plt.plot(v, label=k)
                    (path / f"{algo}").mkdir(exist_ok=True, parents=True)
                    plt.savefig(path / f"{algo}" / f"iter{i}_{k}.pdf")
                    histories[prob_name][algo][k].append(np.stack(v, axis=0))
                metrics = metrics_fn(
                    m_key,
                    target,
                    ids[algo])
                for met_key, met_value in metrics.items():
                    results[prob_name][algo][met_key].append(met_value)

            visualize_key, key = jax.random.split(key, 2)
            visualize(visualize_key,
                      ids,
                      target,
                      path,
                      prefix=f"iter{i}")
    
    #dump results
    def default_to_regular(d):
        if isinstance(d, defaultdict):
            d = {k: default_to_regular(v) for k, v in d.items()}
        return d

    results = default_to_regular(results)
    histories = default_to_regular(histories)
    try:
        with open(parent_path / f'{name}_histories.pkl', 'wb') as f:
            pickle.dump(histories, f)

        with open(parent_path / f'{name}_results.pkl', 'wb') as f:
            pickle.dump(results, f)
    except:
        print("Failed to dump results")

    if compute_metrics:
        for prob_name, problem in PROBLEMS.items():
            for algo in ALGORITHMS:
                for metric_name, run in histories[prob_name][algo].items():
                    run = np.stack(run, axis=0)
                    assert run.shape == (n_rerun, n_updates)
                    last = run[:, -1]
                    if len(histories) > 1:
                        mean = np.mean(last, axis=-1)
                        std = np.std(last, axis=-1) 
                    else:
                        mean = last[0]
                        std = 0
                    print(f"{algo} on {prob_name} with {metric_name} has mean {mean:.3f} and std {std:.3f}")
    
    for prob_name, problem in PROBLEMS.items():
        for algo in ALGORITHMS:
            if algo in results[prob_name].keys():
                for met_name, run in  results[prob_name][algo].items():
                    if len(run) > 1:
                        run = np.stack(run, axis=-1)
                        mean = np.mean(run, axis=-1)
                        std = np.std(run, axis=-1)
                    else:
                        mean = run[0]
                        std = 0
                    print(f"{algo} on {prob_name} {met_name} has mean {mean:.3f} and std {std:.3f}")

if __name__ == "__main__":
    app()