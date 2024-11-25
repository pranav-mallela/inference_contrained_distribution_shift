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

# The observed biased distribution is: 
#          70% hospitalizations are the majority
#          and 30% of hospitalization are the minority
Q = pd.DataFrame({
    'race': np.random.choice(['majority', 'minority'], size=1000, p=[0.7, 0.3]),
    'hospitalized': np.random.choice([0,1], size=1000, p=[0, 1])
})
print(Q)

# Since we only have one covariate we don't need to calcualte the covariance between features
# Our problem is a counting problem 

# strata_dfs = get_feature_strata(Q, target='hospitalized', levels=[['majority', 'minority']])
# strata_counts = get_strata_counts(strata_dfs, levels=[['race'], ['majority', 'minority']])
strata_counts = torch.tensor([560,120])
strata_estimands = {
    "count": strata_counts  # Majority, Minority
}


print("strata counts: ", strata_counts)

feature_weights = torch.eye(2) # one for majority and one for minority

# Our restrictions are that the expectations of the Q should match expectation of P
# Let assume that in P, 
#          the 80% hospitalizations are the majority
#          and 40% of hospitalization are the minority

# I think the restrictions should be that the counts should match ?
restrictions = {
    'majority': 0.7,
    'minority': 0.3
}

# restrictions = SimulationRestrictions(
#     dataset=Q,  # Your dataset object
#     restriction_type='count',  # Or another valid type
#     n_cov_pairs=None  # Optional, depending on your configuration
# )

# def _get_count_restrictions_matrix(
#         self,
#         feature_weights: torch.Tensor,
#         counts_matrix: torch.Tensor,
#         treatment_level: list[str],
#     ) -> tuple[np.ndarray, np.ndarray]:
#         _, features = feature_weights.shape
#         level_size = len(treatment_level)

#         y_0_ground_truth = torch.zeros(level_size, features)
#         y_1_ground_truth = torch.zeros(level_size, features)

#         data_count_0 = counts_matrix[0]
#         data_count_1 = counts_matrix[1]

#         for level in range(level_size):
#             t = data_count_0[level].flatten().unsqueeze(1)
#             features = feature_weights[level * t.shape[0] : (level + 1) * t.shape[0]]
#             y_0_ground_truth[level] = (features * t).sum(dim=0)

#         for level in range(level_size):
#             t_ = data_count_1[level].flatten().unsqueeze(1)
#             features_ = feature_weights[level * t.shape[0] : (level + 1) * t.shape[0]]
#             y_1_ground_truth[level] = (features_ * t_).sum(dim=0)

#         return y_0_ground_truth.numpy(), y_1_ground_truth.numpy()
    
# def get_cvxpy_restrictions(
#         self, cvxpy_weights, feature_weights, ratio, q, n_sample, rho=None
#     ):
#         """
#         Builds constraints in the form needed for the cvxpy package 
#         which is the optimization.
#         """
#         if self.restriction_values is None:
#             raise ValueError("Restriction values not set. Run build_restriction_values")
#         if self.restriction_matrices is None:
#             raise ValueError(
#                 "Restriction matrices not set. Run build_restriction_matrices"
#             )
#         dataset_size = self.dataset.population_df_colinear.shape[0]

#         restrictions = [feature_weights @ cvxpy_weights >= n_sample / dataset_size]
#         if self.restriction_type == "count":
#             restrictions += [
#                 self.restriction_matrices["count"][0] @ cvxpy_weights
#                 == self.restriction_values["count"][0],
#                 self.restriction_matrices["count"][1] @ cvxpy_weights
#                 == self.restriction_values["count"][1],
#             ]
#         return restrictions       
            
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

    alpha = torch.rand(2, requires_grad=True)
    optim = torch.optim.Adam([alpha], 0.01)
    loss_values = []
    # n_sample = data_count_1.sum() + data_count_0.sum()

    sample_size = data_count_0 + data_count_1
    population_size = 1000
    
    data_count = strata_estimands["count"] 
    ideal_distribution = torch.tensor([0.7, 0.3]) 
    
    for index in tqdm(
        range(n_iters), desc=f"Optimizing Upper Bound: {upper_bound}", leave=False
    ):
        w = cp.Variable(2, nonneg=True)  # 2 groups: majority, minority
        # print("Value of w:", w.value)
        alpha_fixed = alpha.squeeze().detach().numpy()
        
        # # Define objective: Minimize difference between weighted expected values and ideal distribution
        # weighted_counts = data_count.detach().numpy()
        # total_weighted_counts = cp.sum(weighted_counts)
        # normalized_counts = weighted_counts / total_weighted_counts  # Proportions

        # # objective = cp.Minimize(
        # #     cp.sum_squares(normalized_counts * w - ideal_distribution.detach().numpy())
        # # )
        objective = cp.sum_squares(w - alpha_fixed)

        # Define constraints
        cvxpy_restrictions = [
            # feature_weights @ w >= sample_size / population_size,
            feature_weights[0] @ w == restrictions["majority"],
            feature_weights[1] @ w == restrictions["minority"],
        ]

        prob = cp.Problem(cp.Minimize(objective), cvxpy_restrictions)
        prob.solve()

        # Handle optimization failure
        if w.value is None:
            print("\nOptimization failed.\n")
            break
        print("Value of w:", w.value)
        alpha.data = torch.tensor(w.value).float()
        # Compute expected values
        expected_values = (alpha * data_count)/sample_size
        print("expected_value", expected_values)
        total_expected = expected_values.sum()
        normalized_expected = expected_values / total_expected
        print("normalized_expected", normalized_expected)
        
        # Compute loss: Squared difference between expected values and ideal distribution
        mse = torch.sum((normalized_expected - ideal_distribution) ** 2)
        # loss_values.append(loss.detach().numpy())

        loss = mse if upper_bound else -mse
        print(loss)
        loss_values.append(loss)
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
# def run_search(
#     restrictions: Dict,
#     strata_estimands: Dict[str, torch.Tensor],
#     feature_weights: torch.Tensor,
#     upper_bound: bool,
#     n_iters: int,
#     rho: Union[float, None],
# ) -> Tuple[float, List[float]]:
#     """Runs the optimization to find the upper or lower bound for the
#     conditional mean.

#     Parameters
#     ----------
#     restrictions : constraints.Restrictions
#         Set of constraints to be used in the optimization.
#     strata_estimands : Dict[str, torch.Tensor]
#         Dictionary with strata counts for each combination of levels.
#     feature_weights : torch.Tensor
#         Tensor that parametrizes the restrictions matrix
#     upper_bound : bool
#         Whether to find the upper or lower bound.
#     n_iters : int
#         Number of iterations to run the optimization for the intervals.
#     rho : Union[float, None]
#         Value of rho to use in the optimization. This is only relevant
#         when calculating bounds with DRO.

#     Returns
#     -------
#     Tuple[float, List[float]]
#         Tuple with the optimized bound value and the history of bound values.
#     """
#     # data_count_0 = strata_estimands["count"][0]
#     # data_count_1 = strata_estimands["count"][1]

#     # alpha = torch.rand(feature_weights.shape[1], requires_grad=True)
#     # optim = torch.optim.Adam([alpha], 0.01)
#     loss_values = []
#     # n_sample = data_count_1.sum() + data_count_0.sum()

#     # sample_size = data_count_0 + data_count_1
#     # population_size = 1000
    
#     data_count = strata_estimands["count"] 
#     ideal_distribution = torch.tensor([0.7, 0.3]) 
    
#     for index in tqdm(
#         range(n_iters), desc=f"Optimizing Upper Bound: {upper_bound}", leave=False
#     ):
#         w = cp.Variable(2, nonneg=True)  # 2 groups: majority, minority
#         print("Value of w:", w.value)
#         # Define objective: Minimize difference between weighted expected values and ideal distribution
#         weighted_counts = data_count.detach().numpy()
#         total_weighted_counts = cp.sum(weighted_counts)
#         normalized_counts = weighted_counts / total_weighted_counts  # Proportions

#         objective = cp.Minimize(
#             cp.sum_squares(normalized_counts * w - ideal_distribution.detach().numpy())
#         )

#         # Define constraints
#         cvxpy_restrictions = [
#             feature_weights[0] @ w == restrictions["majority"],
#             feature_weights[1] @ w == restrictions["minority"],
#         ]

#         prob = cp.Problem(objective, cvxpy_restrictions)
#         prob.solve()

#         # Handle optimization failure
#         if w.value is None:
#             print("\nOptimization failed.\n")
#             break

#         # Compute expected values
#         weights = torch.tensor(w.value, dtype=torch.float32)
#         expected_values = weights * data_count
#         total_expected = expected_values.sum()
#         normalized_expected = expected_values / total_expected

#         # Compute loss: Squared difference between expected values and ideal distribution
#         loss = torch.sum((normalized_expected - ideal_distribution) ** 2)
#         loss_values.append(loss.detach().numpy())

#     final_loss = loss_values[-1] if len(loss_values) > 0 else np.nan
#     return final_loss, loss_values

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