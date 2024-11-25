import argparse
import datetime
import itertools
import json
import os
from typing import Dict, List, Tuple, Union

import constraints
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from config import parse_config_dict
from datasets import FolktablesLoader, SimulationLoader
from tqdm import tqdm
from inference import *
from constraints import *

# Implemnting a dummy experiment
# We only have one covariate - race
# We are going to estimate CATE similar to the paper, so we will assume that those severly infected
# require hospitalization
np.random.seed(42)

# Let assume that in P, 
#          the 80% hospitalizations are the majority
#          and 40% of hospitalization are the minority
P = pd.DataFrame({
    'race': np.random.choice(['majority', 'minority'], size=1000, p=[0.7, 0.3]),
    'hospitalized': np.random.choice([0,1], size=1000, p=[0.6, 0.4])
})

hospitalization_counts = P.groupby(['race', 'hospitalized']).size().reset_index(name='count')

print(hospitalization_counts)
hospitalized_counts_1 = hospitalization_counts[
    hospitalization_counts['hospitalized'] == 1
].sort_values('race')['count'].tolist()
hospitalized_counts_0 = hospitalization_counts[
    hospitalization_counts['hospitalized'] == 0
].sort_values('race')['count'].tolist()

restrictions = {
    'hospitalized': 0.7,
    'not hospitalized': 0.3
}
# The observed biased distribution is: 
#          70% hospitalizations are the majority
#          and 30% of hospitalization are the minority

observed_probs = {
    ('majority', 1): 0.7,  # 70% of hospitalizations are from the majority
    ('minority', 1): 0.3   # 30% of hospitalizations are from the minority
}

# Implement sampling mechanism R and create observed data set Q
def calculate_weight(row):
    if row['hospitalized'] == 1:  # For hospitalized individuals
        return observed_probs[(row['race'], row['hospitalized'])]
    else:  # For non-hospitalized individuals, keep the weight neutral
        return 1.0

P['weight'] = P.apply(calculate_weight, axis=1)

# Sample from P to create Q based on the calculated weights
Q = P.sample(n=1000, weights=P['weight'], replace=True).drop(columns=['weight'])

print(Q)

hospitalization_counts = Q.groupby(['race', 'hospitalized']).size().reset_index(name='count')

print(hospitalization_counts)
hospitalized_counts_1 = hospitalization_counts[
    hospitalization_counts['hospitalized'] == 1
].sort_values('race')['count'].tolist()
hospitalized_counts_0 = hospitalization_counts[
    hospitalization_counts['hospitalized'] == 0
].sort_values('race')['count'].tolist()

strata_counts = [hospitalized_counts_0, hospitalized_counts_1]
print("strata counts: ", strata_counts)
strata_estimands = {
    "count": strata_counts  # Majority, Minority
}

feature_weights = torch.eye(4) # one for majority and one for minority

def _get_count_restrictions(
    self, data: pd.DataFrame, target: str, treatment_level: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes counts of specific outcome 
    (e.g how many belong to hospitalized given that they are a minorty or majority )
    """
    y00_val_1 = sum((data[target] == 0) & (data[treatment_level[0]] == 1))
    y01_val_1 = sum((data[target] == 1) & (data[treatment_level[0]] == 1))

    y00_val_2 = sum((data[target] == 0) & (data[treatment_level[1]] == 1))
    y01_val_2 = sum((data[target] == 1) & (data[treatment_level[1]] == 1))

    restriction_00 = np.array([y00_val_1, y00_val_2])
    restriction_01 = np.array([y01_val_1, y01_val_2])

    return restriction_00, restriction_01

restriction_00, restriction_01 = _get_count_restrictions(Q, P, ['minority', 'majority'])


# Implement the optimization
def run_search(
    restrictions: Dict,
    strata_estimands: Dict[str, torch.Tensor],
    feature_weights: torch.Tensor,
    upper_bound: bool,
    n_iters: int,
    rho: Union[float, None],
) -> Tuple[float, List[float]]:
    """Runs the optimization to find the upper or lower bound for the
    conditional mean.

    Parameters
    ----------
    restrictions : constraints.Restrictions
        Set of constraints to be used in the optimization.
    strata_estimands : Dict[str, torch.Tensor]
        Dictionary with strata counts for each combination of levels.
    feature_weights : torch.Tensor
        Tensor that parametrizes the restrictions matrix
    upper_bound : bool
        Whether to find the upper or lower bound.
    n_iters : int
        Number of iterations to run the optimization for the intervals.
    rho : Union[float, None]
        Value of rho to use in the optimization. This is only relevant
        when calculating bounds with DRO.

    Returns
    -------
    Tuple[float, List[float]]
        Tuple with the optimized bound value and the history of bound values.
    """
    data_count_0 = strata_estimands["count"][0]
    data_count_1 = strata_estimands["count"][1]

    alpha = torch.rand(feature_weights.shape[1], requires_grad=True)
    optim = torch.optim.Adam([alpha], 0.01)
    loss_values = []
    n_sample = data_count_1.sum() + data_count_0.sum()
    population_size = 1000
    print("feature_weights", feature_weights)
    for index in tqdm(
        range(n_iters), desc=f"Optimizing Upper Bound: {upper_bound}", leave=False
    ):
        w = cp.Variable(feature_weights.shape[1])
        alpha_fixed = alpha.squeeze().detach().numpy()

        # ratio = feature_weights @ w
        # q = (data_count_1 + data_count_0) / n_sample
        # q = q.reshape(-1, 1)
        # q = torch.flatten(q)

        objective = cp.sum_squares(w - alpha_fixed)
        # cvxpy_restrictions = restrictions.get_cvxpy_restrictions(
        #     cvxpy_weights=w,
        #     feature_weights=feature_weights,
        #     ratio=ratio,
        #     q=q,
        #     n_sample=n_sample,
        #     rho=rho,
        # )
        
        cvxpy_restrictions = [
            feature_weights @ w >= n_sample / population_size,
            feature_weights @ w == restrictions['not hospitalized'],
            feature_weights @ w == restrictions['hospitalized'],
        ]
        
        print("cvxpy_restrictions: ", cvxpy_restrictions)
        prob = cp.Problem(cp.Minimize(objective), cvxpy_restrictions)
        prob.solve()

        if w.value is None:
            print("\nOptimization failed.\n")
            break
        
        print("Value for W", w.value)
        
        alpha.data = torch.tensor(w.value).float()
        weights_y1 = (feature_weights @ alpha).reshape(*data_count_1.shape)
        weights_y0 = (feature_weights @ alpha).reshape(*data_count_0.shape)
        print("weights_y1: ", weights_y1)
        print("weights_y0: ", weights_y0)
        weighted_counts_1 = weights_y1 * data_count_1
        weighted_counts_0 = weights_y0 * data_count_0
        print("weighted_counts_0: ", weighted_counts_0)
        print("weighted_counts_1: ", weighted_counts_1)

        # ===================================#
        w_counts_1 = weighted_counts_1.select(-1, 1)
        w_counts_0 = weighted_counts_0.select(-1, 1)
        print("w_counts_1: ", w_counts_1)
        print("w_counts_0: ",  w_counts_0)
        # ===================================#

        size = w_counts_1.sum() + w_counts_0.sum()
        conditional_mean = w_counts_1.sum() / size

        print("conditional mean: ", conditional_mean)
        loss = -conditional_mean if upper_bound else conditional_mean
        loss_values.append(conditional_mean.detach().numpy())
        optim.zero_grad()
        loss.backward()
        optim.step()

    ret = np.nan
    if len(loss_values) > 0:
        if upper_bound:
            ret = max(loss_values)
        else:
            ret = min(loss_values)

    return ret, loss_values

max_bound, max_loss_values = run_search(
    restrictions=restrictions,
    strata_estimands=strata_estimands,
    feature_weights=feature_weights,
    upper_bound=True,
    n_iters=10,
    rho=None,
)
min_bound, min_loss_values = run_search(
    restrictions=restrictions,
    strata_estimands=strata_estimands,
    feature_weights=feature_weights,
    upper_bound=False,
    n_iters=10,
    rho=None,
)
print("Bounds [{}, {}]".format(min_bound, max_bound))