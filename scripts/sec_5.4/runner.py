from tqdm import tqdm
from src.id import *
from src.conditional import *
from src.problems.bnn import (
    Protein,
    Concrete,
    Yacht)
from src.trainers.trainer import subsample_trainer
import matplotlib.pyplot as plt
from collections import defaultdict
from src.base import *
from src.utils import (make_step_and_carry,
                       parse_config,
                       config_to_parameters)
from pathlib import Path
from math import ceil
import typer
import wandb


app = typer.Typer()

PROBLEMS = {
    'concrete': Concrete,
    'protein': Protein,
    'yacht': Yacht,
}


def diagnostic(key,
               carry,
               target: Target):
    metrics = {}
    if hasattr(carry, 'id'):
        samples = carry.id.sample(key, 1000, None)
    elif hasattr(carry, 'particles'):
        samples = carry.particles
    else:
        return {}

    assert  np.isnan(samples).sum() == 0
    logp = lambda x: target.log_prob(x,
                                    np.arange(target.train_size),
                                    train=True)
    train_log_probs = jax.vmap(logp, (0))(samples).mean(0)
    train_rmse = vmap(lambda x : target.mse(x, train=True))(samples).mean() ** 0.5
    test_rmse = vmap(lambda x : target.mse(x, train=False))(samples).mean() ** 0.5

    metrics =  {'train_logp': train_log_probs,
                'test_mse': test_rmse,
                'train_mse': train_rmse}
    wandb.log(metrics)

    if hasattr(carry, 'id'):
        samples = carry.id.sample(key, 10000, None)
    else:
        samples = carry.particles
    w1, w2, b1, b2 = jax.vmap(target.bnn.get_w_b)(samples)
    norm = np.linalg.norm
    hist = {}
    hist['w1_norms'] = wandb.Histogram(norm(w1, axis=1))
    hist['w2_norms'] = wandb.Histogram(norm(w2, axis=1))
    hist['b1_norms'] = wandb.Histogram(norm(b1, axis=1))
    hist['b2_norms'] = wandb.Histogram(norm(b2, axis=1))
    wandb.log(hist)
    return metrics


@app.command()
def run_trials(config_name: str,
               algorithm_name: str):
    config_path = Path(f"scripts/sec_5.4/config/{config_name}.yaml")
    parent_path = Path("output/sec_5.4")
    assert config_path.exists(), f"{config_path} does not exist"
    config = parse_config(config_path)
    parameters = config_to_parameters(config, algorithm_name)
    
    n_subsample = config['experiment']['n_subsample']
    problem_name = config['experiment']['problem_name']
    seed = config['experiment']['seed']
    n_reruns = config['experiment']['n_reruns']
    n_updates = config['experiment']['n_updates']
    use_jit = config['experiment']['use_jit']
    
    key = jax.random.PRNGKey(seed)
    histories = defaultdict(list)
    

    pbar = tqdm(range(n_reruns))
    pbar.set_description(f"Subsample {n_subsample} on {problem_name}")

    # Defined Path
    path = parent_path / f"{problem_name}" / f"{n_subsample}"
    path.mkdir(parents=True, exist_ok=True)

    for i in pbar:
        run = wandb.init(project=f"bnn-{config_name}",
                        config={**config[algorithm_name]},
                        reinit=True)
        trainer_key, init_key, eval_key, key = jax.random.split(key, 4)
        target = PROBLEMS[problem_name](seed=seed+i)
        max_epoch = ceil(n_updates /  (target.train_size / n_subsample))
            
        step, carry = make_step_and_carry(
            init_key,
            parameters,
            target
        )

        init_metrics = diagnostic(eval_key,
                                  carry,
                                  target)

        history, carry = subsample_trainer(
            trainer_key,
            carry,
            target,
            None,
            step,
            max_epoch,
            n_subsample,
            metrics=diagnostic,
            use_jit=use_jit,
        )

        save_path = path / f"{algorithm_name}"
        save_path.mkdir(parents=True, exist_ok=True)
        for metric_name, v in history.items():
            plt.clf()
            plt.plot(v)
            plt.savefig(save_path / f"{i}_{metric_name}.pdf")
        zip_dict = lambda dict1, dict2 : {k: (dict1.get(k), dict2.get(k)) for k in set(dict1) | set(dict2)}

        for metric_name, val in zip_dict(dict(init_metrics), dict(history)).items():
            (first, after) = val[0], val[1]
            if first is None:
                res = after
            else:
                res = np.concatenate([np.array([first]), after])
            histories[metric_name].append(res)

    ## Print Summary
    plt.clf()
    for metric, v in histories.items():
        run_results = np.stack(v, axis=0)
        mu, std = np.mean(run_results, axis=0), np.var(run_results, axis=0) ** 0.5
        plt.errorbar(target.train_size / int(n_subsample) * np.arange(mu.shape[0]),
                        mu,
                        yerr=std,
                        label=f"{n_subsample}")
        print(f"{n_subsample} {algorithm_name} on {problem_name} of {metric}"
                + f" has mean {mu[-1]:.3f} and std {std[-1]:.3f}")
    path = parent_path / f"{problem_name}"/ f"{algorithm_name}"
    path.mkdir(parents=True, exist_ok=True)
    plt.legend()
    plt.savefig(path / f"{metric_name}_summary.pdf")
            

if __name__ == "__main__":
    app()