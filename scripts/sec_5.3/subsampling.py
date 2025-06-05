from tqdm import tqdm
import typer
from src.problems.bnn import (Concrete,
                              Protein,
                              Boston,
                              Yacht,
                              SubsampleTest)
from src.id import *
from src.conditional import *
from src.trainers.trainer import subsample_trainer
import matplotlib.pyplot as plt
from collections import defaultdict
from src.base import *
from src.utils import make_step_and_carry
from pathlib import Path
from math import ceil

app = typer.Typer()

SUBSAMPLES = [10, 50, 100]
N_UPDATES = 100
N_RERUN = 10
THETA_LR =  1e-3
THETA_DECAY = 1e-3
DIM_Z = 10
USE_SCHEDULER = True

pvi_parameters = Parameters(
        algorithm='pvi',
        model_parameters=ModelParameters(n_hidden=512,
                                         use_particles=True,
                                         n_particles=100,
                                         d_z=DIM_Z),
        theta_opt_parameters=ThetaOptParameters(lr=THETA_LR,
                                                optimizer='adam',
                                                lr_decay=USE_SCHEDULER,
                                                min_lr=1e-5,
                                                interval=10,
                                                regularization=THETA_DECAY,
                                                clip=False),
        r_opt_parameters=ROptParameters(lr=1e-2,
                                        clip=True,
                                        max_clip=1.,
                                        regularization=1e-2),
        extra_alg_parameters=PIDParameters(0.0),)
svi_parameters = Parameters(
        algorithm='svi',
        model_parameters=ModelParameters(n_hidden=512,
                                         use_particles=False,
                                         d_z=DIM_Z),
        theta_opt_parameters=ThetaOptParameters(lr=THETA_LR,
                                                optimizer='adam',
                                                lr_decay=USE_SCHEDULER,
                                                min_lr=1e-4,
                                                regularization=THETA_DECAY,
                                                interval=10,
                                                clip=False),
        extra_alg_parameters=SVIParameters(),)

ALGORITHMS = {
    'pvi': pvi_parameters,
    'svi': svi_parameters,
}

USE_JIT = True # Set to False for debugging
PROBLEMS = {
    'subsample': SubsampleTest,
    #'protein': Protein,
    #'boston': Boston,
    #'yacht': Yacht,
    #'concrete': Concrete,
}

def diagnostic(key,
               id: ID,
               target: Target):
    samples = id.sample(key, 1000, None)
    assert  np.isnan(samples).sum() == 0
    logp = lambda x: target.log_prob(x,
                                     np.arange(target.train_size),
                                     train=True)
    train_log_probs = jax.vmap(logp, (0))(samples).mean(0)
    train_rmse = vmap(lambda x : target.mse(x, train=True))(samples).mean() ** 0.5
    test_rmse = vmap(lambda x : target.mse(x, train=False))(samples).mean() ** 0.5
    return {'train_logp': train_log_probs,
            'test_mse': train_rmse,
            'train_mse': test_rmse}


@app.command()
def run(
    n_rerun: int=N_RERUN,
    seed: int=2):
    parent_path = Path("output/bnn")
    key = jax.random.PRNGKey(seed)
    histories = defaultdict(
        lambda : defaultdict(
            lambda : defaultdict(
                lambda : defaultdict(list)
                )
        )
    )
    
    for subsample in SUBSAMPLES:
        for prob_name, problem in PROBLEMS.items():
            pbar = tqdm(range(n_rerun))
            pbar.set_description(f"Subsample {subsample} on {prob_name}")
            path = parent_path / f"{prob_name}" / f"{subsample}"
            for i in pbar:
                trainer_key, init_key, eval_key, key = jax.random.split(key, 4)
                target = problem()
                max_epoch = ceil(N_UPDATES /  (target.train_size / subsample))
                path.mkdir(parents=True, exist_ok=True)
                for algo, parameters in ALGORITHMS.items():
                    step, carry = make_step_and_carry(
                        init_key,
                        parameters,
                        target
                    )
                    init_metrics = diagnostic(eval_key,
                                              carry.id,
                                              target)

                    history, carry = subsample_trainer(
                        trainer_key,
                        carry,
                        target,
                        None,
                        step,
                        max_epoch,
                        subsample,
                        metrics=diagnostic,
                        use_jit=USE_JIT,
                    )
                    save_path = path / f"{algo}"
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
                        histories[prob_name][algo][metric_name][subsample].append(res)

            ## Print Summary average run
            for algo in ALGORITHMS:
                for metric_name, subsample_dict in histories[prob_name][algo].items():
                    v = subsample_dict[subsample]
                    run_results = np.stack(v, axis=0)
                    mu, var = np.mean(run_results, axis=0), np.var(run_results, axis=0)
                    plt.clf()
                    plt.errorbar(range(mu.shape[0]), mu, yerr=var**0.5)
                    save_path = path / f"{algo}"
                    plt.savefig(save_path / f"summary_{metric_name}.pdf")
                    plt.savefig(path / f"{algo}_summary_{metric_name}.pdf")


    ## Print Summary
    for prob_name, problem in PROBLEMS.items():
        for algo in ALGORITHMS:
            for metric_name, subsample_dict in histories[prob_name][algo].items():
                plt.clf()
                for subsample_key, v in subsample_dict.items():
                    run_results = np.stack(v, axis=0)
                    mu, std = np.mean(run_results, axis=0), np.var(run_results, axis=0) ** 0.5
                    plt.errorbar(target.train_size / int(subsample_key) * np.arange(mu.shape[0]),
                                 mu,
                                 yerr=std,
                                 label=f"{subsample_key}")
                    print(f"{subsample_key} {algo} on {prob_name} of {metric_name}"
                          + f" has mean {mu[-1]:.3f} and std {std[-1]:.3f}")
                path = parent_path / f"{prob_name}"/ f"{algo}"
                path.mkdir(parents=True, exist_ok=True)
                plt.legend()
                plt.savefig(path / f"{metric_name}_summary.pdf")
            

if __name__ == "__main__":
    app()