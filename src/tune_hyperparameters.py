from joblib import Parallel, delayed
from scipy.stats import uniform, loguniform, randint
from sklearn.model_selection import ParameterSampler
from OGD import ECI_Predictor
import numpy as np
import json
import os

ALPHA = 0.1
TARGET_COVERAGE = 1 - ALPHA
COVERAGE_TOLERANCE = 0.02

def evaluate_single_config(params, predictions, actuals):
    predictor = ECI_Predictor(
        alpha=ALPHA,
        eta=params['eta'],
        q_init=params['q_init'],
        c=params['c'],
        eq_function=params['eq_function'],
        version=params['version'],
        window_length=params.get('window_length', 50),
        h=params.get('h', 1.0)
    )

    coverage_list = []
    width_list = []

    for t in range(len(predictions)):
        y_pred = predictions[t]
        y_true = actuals[t]

        lower, upper = predictor.get_interval(y_pred)
        width_list.append(upper - lower)

        covered = predictor.update(y_pred, y_true)
        coverage_list.append(covered)

    coverage_rate = np.mean(coverage_list)
    mean_width = np.mean(width_list)

    return {
        'coverage_rate': coverage_rate,
        'mean_width': mean_width,
        'params': params
    }

def round_params(params):
    """Round continuous hyperparameters to 2 decimals"""
    rounded = params.copy()
    rounded['eta'] = round(params['eta'], 2)
    rounded['q_init'] = round(params['q_init'], 2)
    rounded['c'] = round(params['c'], 2)
    if 'h' in params:
        rounded['h'] = round(params['h'], 2)
    return rounded

def tune_hyperparameters(predictions, actuals, n_trials=10000, save_path='tuned_params.json'):
    """
    Tune hyperparameters for ECI predictor and save to JSON file.

    Parameters:
    -----------
    predictions : array-like
        Point predictions
    actuals : array-like
        True values
    n_trials : int
        Number of random search trials (default: 10000)
    save_path : str
        Path to save the tuned parameters (default: 'tuned_params.json')

    Returns:
    --------
    best_params_basic : dict
        Best parameters for basic variant
    best_params_cutoff : dict
        Best parameters for cutoff variant
    """

    # Define parameter distributions for random search
    param_distributions_basic = {
        'eta': loguniform(0.01, 10.0),
        'q_init': uniform(5.0, 100.0),
        'c': loguniform(0.1, 10.0),
        'eq_function': ['sigmoid', 'gaussian'],
        'version': ['basic']
    }

    param_distributions_cutoff = {
        'eta': loguniform(0.01, 10.0),
        'q_init': uniform(5.0, 100.0),
        'c': loguniform(0.1, 10.0),
        'eq_function': ['sigmoid', 'gaussian'],
        'window_length': randint(20, 201),  # 20 to 200
        'h': uniform(0.3, 2.0),
        'version': ['cutoff']
    }

    configs_basic = list(ParameterSampler(param_distributions_basic, n_iter=n_trials//2, random_state=42))
    configs_cutoff = list(ParameterSampler(param_distributions_cutoff, n_iter=n_trials//2, random_state=43))
    all_configs = configs_basic + configs_cutoff

    results = Parallel(n_jobs=-1)(
        delayed(evaluate_single_config)(config, predictions, actuals)
        for config in all_configs
    )

    valid_results = [r for r in results
                     if abs(r['coverage_rate'] - TARGET_COVERAGE) <= COVERAGE_TOLERANCE]

    valid_basic = [r for r in valid_results if r['params']['version'] == 'basic']
    valid_cutoff = [r for r in valid_results if r['params']['version'] == 'cutoff']

    best_basic = min(valid_basic, key=lambda x: x['mean_width'])
    best_params_basic_rounded = round_params(best_basic['params'])
    print("\nBEST BASIC:")
    print(f"  Coverage Rate: {best_basic['coverage_rate']:.3f} (Target: {TARGET_COVERAGE:.3f})")
    print(f"  Mean Width: {best_basic['mean_width']:.2f}")
    print(f"  Params: {best_params_basic_rounded}")
    best_params_basic = best_params_basic_rounded

    best_cutoff = min(valid_cutoff, key=lambda x: x['mean_width'])
    best_params_cutoff_rounded = round_params(best_cutoff['params'])
    print("\nBEST CUTOFF:")
    print(f"  Coverage Rate: {best_cutoff['coverage_rate']:.3f} (Target: {TARGET_COVERAGE:.3f})")
    print(f"  Mean Width: {best_cutoff['mean_width']:.2f}")
    print(f"  Params: {best_params_cutoff_rounded}")
    best_params_cutoff = best_params_cutoff_rounded

    # Save to JSON
    tuned_params = {
        'basic': best_params_basic,
        'cutoff': best_params_cutoff
    }

    with open(save_path, 'w') as f:
        json.dump(tuned_params, f, indent=2)

    print(f"\nSaved tuned parameters to {save_path}")

    return best_params_basic, best_params_cutoff
