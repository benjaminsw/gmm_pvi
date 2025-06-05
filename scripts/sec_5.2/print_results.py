from pathlib import Path
import numpy as np
import pickle
import typer 
from src.problems.toy import *
from toy_de import metrics_fn

PROBLEMS = {
    'banana': Banana,
    'multimodal': Multimodal,
    'xshape': XShape,
}

app = typer.Typer()

@app.command()
def baseline(seed: int = 1):
    key = jax.random.PRNGKey(seed)
    for prob_name, problem in PROBLEMS.items():
        w_key, key = jax.random.split(key, 2)
        target = problem()
        met = metrics_fn(
            w_key, 
            target,
            target)
        for met_name, met_value in met.items():
            print(f"{prob_name} {met_name}: {met_value:.3f}")

@app.command()
def results(name: str, algo:str):
    path = Path(f"output/toy_de/{name}")
    assert path.is_dir(), f"Path {path.absolute()} does not exist"
    results = path / f'{name}_results.pkl'

    with open(results, 'rb') as f:
        results = pickle.load(f)

    for prob_name, problem in results.items():
        assert algo in problem.keys(), f"Algorithm {algo} is not in the results"
        run = problem[algo]
        for met_name, run in run.items():
            if len(run) > 1:
                run = np.stack(run, axis=-1)
                mean = np.mean(run, axis=-1)
                std = np.std(run, axis=-1)
            else:
                mean = run[0]
                std = 0
            print(f"{algo} on {prob_name} {met_name} has mean {mean:.2f} and std {std:.2f}")


if __name__ == "__main__":
    app()