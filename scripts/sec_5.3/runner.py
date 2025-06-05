from tqdm import tqdm
from src.id import *
from src.conditional import *
from src.problems.logistic_regression import Waveform
from src.trainers.trainer import subsample_trainer
import matplotlib.pyplot as plt
from src.base import *
from src.utils import (make_step_and_carry,
                       parse_config,
                       config_to_parameters)
from pathlib import Path
from math import ceil
import typer


app = typer.Typer()

PROBLEMS = {
    'waveform': Waveform,
}


@app.command()
def run_trials(config_name: str,
               algorithm_name: str):
    config_path = Path(f"scripts/sec_5.3/config/{config_name}.yaml")
    parent_path = Path("output/sec_5.3")
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

    pbar = tqdm(range(n_reruns))
    pbar.set_description(f"Subsample {n_subsample} on {problem_name}")

    # Defined Path
    path = parent_path / f"{problem_name}" / f"{n_subsample}"
    path.mkdir(parents=True, exist_ok=True)

    for i in pbar:
        trainer_key, init_key, key = jax.random.split(key, 3)
        target = PROBLEMS[problem_name]()
        max_epoch = ceil(n_updates /  (target.train_size / n_subsample))
            
        step, carry = make_step_and_carry(
            init_key,
            parameters,
            target
        )

        history, carry = subsample_trainer(
            trainer_key,
            carry,
            target,
            None,
            step,
            max_epoch,
            n_subsample,
            metrics=None,
            use_jit=use_jit,
        )

        save_path = path / f"{algorithm_name}"
        save_path.mkdir(parents=True, exist_ok=True)
        for metric_name, v in history.items():
            plt.clf()
            plt.plot(v)
            plt.savefig(save_path / f"{i}_{metric_name}.pdf")
        samples = carry.id.sample(key, 1000, None)
        np.save(save_path / f"{i}_samples.npy", samples)
        


if __name__ == "__main__":
    app()