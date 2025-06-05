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
from sklearn.neighbors import KernelDensity


app = typer.Typer()
 

colors = ['deepskyblue', 'red']
problem = Bimodal
means = [1, 2, 4]

def no_bigger_than(x, nums):
    acc = 0 
    for num in nums:
        acc += x >= num
    return acc

def rescale(x):
    return  (x ** 2) * 0.8

alpha_value = lambda mu: rescale(no_bigger_than(mu, means) / len(means))


def visualize_means(idss,
                   path,
                   prefix=""):
    _max = 8
    _min = -8
    x_lin = np.linspace(_min, _max, 20)
    XX, YY = np.meshgrid(x_lin, x_lin)
    XY = np.stack([XX, YY], axis=-1)

    plt.clf()
    for mu, ids in idss.items():
        for i, (name, id) in enumerate(ids.items()):
            if hasattr(id.conditional, 'net'):
                plt.clf()
                f = lambda z: id.conditional.net(z, None)
                means = vmap(vmap(f))(XY)[0]
                max_norm = np.max(np.linalg.norm(means, axis=-1))
                norm_means = means / max_norm 
                plt.quiver(XX, YY, norm_means[:, :, 0], norm_means[:, :, 1])
                plt.scatter(id.particles[:, 0], id.particles[:, 1], color='black', alpha=0.5)
                plt.savefig(path / f"{name}_{mu}_means.pdf")

def visualize_grad(idss,
                   path,
                   prefix=""):
    _max = 8
    _min = -8
    x_lin = np.linspace(_min, _max, 1000)
    XX, YY = np.meshgrid(x_lin, x_lin)
    XY = np.stack([XX, YY], axis=-1)

    plt.clf()
    bw = 1.
    for mu, ids in idss.items():
        for i, (name, id) in enumerate(ids.items()):
            if hasattr(id.conditional, 'net'):
                plt.clf()
                f = lambda z: id.conditional.net(z, None)
                grad = vmap(vmap(jax.jacfwd(f)))(XY)[0]
                largest_eigh = lambda m : jax.lax.linalg.eigh(m @ m.T, sort_eigenvalues=True)[1][-1] ** 0.5
                norm = vmap(vmap(largest_eigh))(grad)
                #norm = np.linalg.matrix_norm(grad)
                plt.contourf(XX, YY, norm)
                plt.scatter(id.particles[:, 0], id.particles[:, 1], color='black', alpha=0.5)
                plt.colorbar()
                plt.savefig(path / f"{name}_{mu}_grad_norm.pdf")

def visualize_particles_pdf(
        idss,
        path,
        prefix=""):
    _max = 8
    _min = -8
    x_lin = np.linspace(_min, _max, 1000)
    XX, YY = np.meshgrid(x_lin, x_lin)
    XY = np.stack([XX, YY], axis=-1)

    plt.clf()
    bw = 1.
    for mu, ids in idss.items():
        for i, (name, id) in enumerate(ids.items()):
            if name == 'pvi':
                kernel = KernelDensity(kernel='gaussian', bandwidth=bw).fit(id.particles)
                f = np.reshape(np.exp(kernel.score_samples(XY.reshape(-1, 2))), XX.shape)
                # Extract the KDE data
                cp = plt.contour(
                    XX, YY, f,
                    levels= 5, #[0.1, 0.5, 0.8],
                    linewidths=1.5,
                    colors=colors[i],
                    alpha=alpha_value(mu))
        plt.xlim(_min, _max)
        plt.ylim(_min, _max)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(path / f"{prefix}_particles_kde.pdf")


def visualize_particles(
        idss,
        path,
        prefix=""):
    _max = 8
    _min = -8
    for mu, ids in idss.items():
        plt.clf()
        for i, (name, id) in enumerate(ids.items()):
            c_model = plt.scatter(
                id.particles[:, 0],
                id.particles[:, 1],
                color=colors[i],
                linewidths=1.5,
                alpha=alpha_value(mu),
                label=name)
        plt.xlim(_min, _max)
        plt.ylim(_min, _max)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path / f"{prefix}_{mu}_particles.pdf")

'''
def visualize_particles(
        idss,
        path,
        prefix=""):
    _max = 8
    _min = -8
    plt.clf()
    for mu, ids in idss.items():
        target = problem(mu)

        for i, (name, id) in enumerate(ids.items()):
            colors = ['red', 'blue']
            c_model = plt.scatter(
                id.particles[:, 0],
                id.particles[:, 1],
                color=colors[i],
                linewidths=1.5,
                alpha=alpha_value(mu),
                label=name)
    plt.xlim(_min, _max)
    plt.ylim(_min, _max)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(path / f"{prefix}_particles.pdf")
'''

def visualize_pdf(
        idss,
        path,
        prefix=""):
    _max = 8
    _min = -8
    x_lin = np.linspace(_min, _max, 1000)
    XX, YY = np.meshgrid(x_lin, x_lin)
    XY = np.stack([XX, YY], axis=-1)
    plt.clf()
    for mu, ids in idss.items():
        target = problem(mu)
        log_p = lambda x : target.log_prob(x, None)
        log_true_ZZ = vmap(vmap(log_p))(XY)

        c_true = plt.contour(
            XX,
            YY,
            np.exp(log_true_ZZ),
            levels=5,
            colors='black',
            linewidths=3,
            alpha=alpha_value(mu))

        cs = {}
        for i, (name, id) in enumerate(ids.items()):
            model_log_p = lambda x: id.log_prob(x, None)
            log_model_ZZ = vmap(vmap(model_log_p))(XY)
            cs[name] = plt.contour(
                XX,
                YY,
                np.exp(log_model_ZZ),
                levels=c_true._levels,
                colors=colors[i],
                linewidths=1.5,
                alpha=alpha_value(mu))
    labels = ['True', 'PVI', 'PVI Zero']
    cs_color = [c.collections[-1].get_edgecolor() for c in [c_true, *cs.values()]]
    lines = [plt.Line2D([0],
                        [0],
                        color=color,
                        lw=2) for color in cs_color]
    plt.legend(lines, labels)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(_min, _max)
    plt.ylim(_min, _max)
    plt.savefig(path / f"{prefix}_pdf.pdf")


def visualize(
        idss,
        path,
        prefix=""):
    visualize_means(idss, path, prefix)
    visualize_particles(idss, path, prefix)
    visualize_pdf(idss, path, prefix)
    visualize_particles_pdf(idss, path, prefix)
    #visualize_grad(idss, path, prefix)

@app.command()
def run(config_name: str,
        seed: int=2):
    config_path = Path(f"scripts/sec_5.1/config/{config_name}.yaml")
    assert config_path.exists()
    config = parse_config(config_path)

    n_rerun = config['experiment']['n_reruns']
    n_updates = config['experiment']['n_updates']
    name = config['experiment']['name']
    name = 'default' if len(name) == 0 else name
    compute_metrics = config['experiment']['compute_metrics']
    use_jit = config['experiment']['use_jit']
    run_zero = config['experiment']['run_zero']

    parent_path = Path(f"output/sec_5.1/{name}")
    key = jax.random.PRNGKey(seed)
    histories = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
    results = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))

    trainer_key, init_key, key = jax.random.split(key, 3)
    idss = {}
    path = parent_path / f"bimodal"
    path.mkdir(parents=True, exist_ok=True)


    for mean in means:
        target = problem(mean)
        ids = {}
        will_run_zero = 2 if run_zero else 1
        for j in range(will_run_zero):
            parameters = config_to_parameters(config, 'pvi')
            name = 'pvi_zero' if j == 1 else 'pvi'
            if j == 1:
                parameters = parameters._replace(
                    r_opt_parameters=parameters.r_opt_parameters._replace(lr=0.)
                )
            step, carry = make_step_and_carry(
                init_key,
                parameters,
                target)
            
            history, carry = trainer(
                trainer_key,
                carry,
                target,
                None,
                step,
                n_updates,
                metrics=None,
                use_jit=use_jit,
            )
            ids[name] = carry.id
        idss[mean] = ids
    visualize(
            idss,
            path,
            prefix=None)
    

if __name__ == "__main__":
    app()