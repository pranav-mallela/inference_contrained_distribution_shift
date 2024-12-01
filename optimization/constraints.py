import itertools
from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Tuple, Union

import cvxpy as cp
import numpy as np
import pandas as pd
import torch


class Parametrization(ABC):
    @abstractmethod
    def get_feature_weights(self, levels: list[list[str]]) -> torch.Tensor:
        pass


class Restrictions(ABC):
    @abstractmethod
    def build_restriction_values(self):
        pass

    @abstractmethod
    def build_restriction_matrices(self, feature_weights, strata_estimands):
        pass

    @abstractmethod
    def get_cvxpy_restrictions(
        self, cvxpy_weights, feature_weights, ratio, q, n_sample, rho=None
    ):
        pass

# this is the class that we are using
class SimulationParametrizationOurs(Parametrization):
    def __init__(self, matrix_type: Literal["unrestricted", "separable", "targeted"]):
        if matrix_type not in ["unrestricted", "separable", "targeted"]:
            raise ValueError(
                "matrix_type must be one of 'unrestricted', 'separable', 'targeted'"
            )
        self.matrix_type = matrix_type

    def get_feature_weights(
        self, *_: list[list[str]], **__: Callable
    ) -> Tuple[torch.Tensor, Union[dict, None]]:
        if self.matrix_type == "unrestricted":
            print("Feature weights: ", torch.eye(2, 2))
            # return torch.eye(2, 2), None
            # return torch.eye(4, 4), None #this for [2]
            return torch.eye(12, 12), None #this for features [2] [3] [2]
        elif self.matrix_type == "separable":
            pass
        elif self.matrix_type == "targeted":
            pass
        
class SimulationParametrization(Parametrization):
    def __init__(self, matrix_type: Literal["unrestricted", "separable", "targeted"]):
        if matrix_type not in ["unrestricted", "separable", "targeted"]:
            raise ValueError(
                "matrix_type must be one of 'unrestricted', 'separable', 'targeted'"
            )
        self.matrix_type = matrix_type

    def get_feature_weights(
        self, *_: list[list[str]], **__: Callable
    ) -> Tuple[torch.Tensor, Union[dict, None]]:
        if self.matrix_type == "unrestricted":
            return torch.eye(12, 12), None
        elif self.matrix_type == "separable":
            block_1 = torch.eye(6, 6)
            block_1[:, -2] = 1
            block_1[:, -1] = 0

            block_2 = torch.eye(6, 6)
            block_2[:, -2] = 0
            block_2[:, -1] = 1
            return torch.cat([block_1, block_2]), None
        elif self.matrix_type == "targeted":
            return torch.cat([torch.eye(6, 6), torch.eye(6, 6)]), None


class SemiSyntheticParametrization(Parametrization):
    def __init__(
        self,
        matrix_type: Literal["unrestricted", "separable", "targeted"],
    ):
        if matrix_type not in ["unrestricted", "separable", "targeted"]:
            raise ValueError(
                "matrix_type must be one of 'unrestricted', 'separable', 'targeted'"
            )
        self.matrix_type = matrix_type

    def get_feature_weights(
        self, levels: list[list[str]]
    ) -> Tuple[torch.Tensor, Union[dict, None]]:
        number_strata = 1
        for level in levels:
            number_strata *= len(level)

        degrees_of_freedom = len(levels[1]) * len(levels[2])
        if self.matrix_type == "unrestricted":
            feature_weights = torch.eye(number_strata)
            idx = 0
            hash_map = {}
            for combination in self._traverse_level_combinations(levels):
                hash_map[combination] = idx
                idx += 1
            return feature_weights, hash_map
        elif self.matrix_type == "separable":
            idx = 0
            idj = 0
            idm = 0
            feature_weights = torch.zeros(number_strata, degrees_of_freedom + 2)
            starting_tuple = (levels[1][0], levels[2][0])
            previous_tuple = starting_tuple
            starting_feature = "white"
            previous_feature = starting_feature
            for combination in self._traverse_level_combinations(levels):
                current_tuple = (combination[1], combination[2])
                current_feature = combination[0]
                if previous_tuple != current_tuple:
                    if current_tuple == starting_tuple:
                        idj = 0
                    else:
                        idj += 1
                if previous_feature != current_feature:
                    if current_feature == starting_feature:
                        idm = 0
                    else:
                        idm += 1

                weight = [0] * degrees_of_freedom
                weight[idj] = 1
                weight_race = [0] * 2
                weight_race[idm] = 1
                feature_weights[idx] = torch.tensor(weight + weight_race).float()
                idx += 1
                previous_tuple = current_tuple
                previous_feature = current_feature
            return feature_weights, None
        elif self.matrix_type == "targeted":
            idx = 0
            idj = 0
            feature_weights = torch.zeros(number_strata, degrees_of_freedom)
            starting_tuple = (levels[1][0], levels[2][0])
            previous_tuple = starting_tuple
            hash_map = {}
            for combination in self._traverse_level_combinations(levels):
                current_tuple = (combination[1], combination[2])
                if previous_tuple != current_tuple:
                    if current_tuple == starting_tuple:
                        idj = 0
                    else:
                        idj += 1
                weight = [0] * degrees_of_freedom
                weight[idj] = 1
                feature_weights[idx] = torch.tensor(weight).float()
                hash_map[current_tuple] = idx
                idx += 1
                previous_tuple = current_tuple
            return feature_weights, hash_map

    def _traverse_level_combinations(self, levels: list[list[str]]) -> list[tuple]:
        """
        Iterates over all combinations of levels for stratification
        """
        for combination in itertools.product(*levels):
            yield combination


class SimulationRestrictions(Restrictions):
    def __init__(
        self, dataset, restriction_type: str, n_cov_pairs: Optional[int] = None
    ):
        self.restriction_type = restriction_type
        self.n_cov_pairs = n_cov_pairs
        self.dataset = dataset
        self.restriction_values = None
        self.restriction_matrices = None
        self.all_feature_means = None

    def _get_count_restrictions(
        self, data: pd.DataFrame, target: str, treatment_level: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes counts of specific outcome 
        (e.g how many belong to hospitalized given that they are a minorty or majority )
        """
        y00_val_1 = sum((data[target] == 0) & (data[treatment_level[0]] == 1)) # not hopsitalized and majority
        y01_val_1 = sum((data[target] == 1) & (data[treatment_level[0]] == 1)) # hopsitalized and majority

        y00_val_2 = sum((data[target] == 0) & (data[treatment_level[1]] == 1))
        y01_val_2 = sum((data[target] == 1) & (data[treatment_level[1]] == 1))

        restriction_00 = np.array([y00_val_1, y00_val_2])
        restriction_01 = np.array([y01_val_1, y01_val_2])

        print("Restriction Sizes: ", restriction_00.shape, restriction_01.shape)
        return restriction_00, restriction_01

    def build_restriction_values(self) -> None:
        restriction_values = {
            "count": self._get_count_restrictions(
                data=self.dataset.population_df_colinear,
                target=self.dataset.target,
                treatment_level=self.dataset.levels_colinear[0],
            ),
            "count_minus": self._get_count_restrictions(
                data=self.dataset.population_df_colinear,
                target=self.dataset.target,
                treatment_level=self.dataset.levels_colinear[0],
            ),
            "count_plus": self._get_count_restrictions(
                data=self.dataset.population_df_colinear,
                target=self.dataset.target,
                treatment_level=self.dataset.levels_colinear[0],
            )
            + self._get_count_restrictions(
                data=self.dataset.population_df_colinear,
                target=self.dataset.alternate_outcome,
                treatment_level=self.dataset.levels_colinear[0],
            ),
        }
        print("Built restriction_values: ", restriction_values)

        if self.n_cov_pairs and self.restriction_type.startswith("cov"):
            all_cov_vars = self._get_cov_pairs(
                n_pairs=self.n_cov_pairs,
                mode=self.restriction_type.split("_")[1],
            )
            restriction_values["cov"] = self._get_cov_restrictions(
                data=self.dataset.population_df_colinear,
                target=self.dataset.target,
                treatment_level=self.dataset.levels_colinear[0],
                all_cov_vars=all_cov_vars,
            )

            all_feature_means = []
            for cov_pair in all_cov_vars:
                feature_means = {
                    cov_pair[0]: self.dataset.population_df_colinear[
                        cov_pair[0]
                    ].mean(),
                    cov_pair[1]: self.dataset.population_df_colinear[
                        cov_pair[1]
                    ].mean(),
                }
                all_feature_means.append(feature_means)
            self.all_feature_means = all_feature_means

        self.restriction_values = restriction_values

    def _get_cov_pairs(self, n_pairs, mode):
        input_df = self.dataset.population_df_colinear.copy(())
        all_cols = input_df.columns.to_list()
        first_group_prefix = all_cols[0].split("_")[0]

        first_group_cols = input_df.filter(
            regex=f"^{first_group_prefix}_"
        ).columns.to_list()
        excluded_cols = (
            first_group_cols + [self.dataset.target] + self.dataset.levels_colinear[0]
        )
        selected_cols = [col for col in all_cols if col not in excluded_cols]
        covs = input_df.cov().loc[selected_cols, first_group_cols].unstack()

        if mode == "positive":
            covs = covs[covs >= 0]
            cov_pairs = covs.sort_values(ascending=False).index.to_list()[:n_pairs]

        elif mode == "negative":
            covs = covs[covs <= 0]
            cov_pairs = covs.sort_values(ascending=True).index.to_list()[:n_pairs]

        elif mode == "strict":
            cov_pairs = covs.sort_values(ascending=False).index.to_list()[:n_pairs]
        return cov_pairs

    def _get_cov_restrictions_matrix(
        feature_weights: torch.Tensor,
        all_covs: torch.Tensor,
        treatment_level: list[str],
    ):
        """
        Calculates covariance-based restrictions between features
        (In our first case we only have race so this isn't needed)
        """
        _, features = feature_weights.shape
        level_size = len(treatment_level)

        y_0_ground_truth = torch.zeros(level_size, features)
        y_1_ground_truth = torch.zeros(level_size, features)

        all_matrices = []
        for covs in all_covs:
            data_covs_0 = covs[0]
            data_covs_1 = covs[1]

            for level in range(level_size):
                t = data_covs_0[level].flatten().unsqueeze(1)
                features = feature_weights[
                    level * t.shape[0] : (level + 1) * t.shape[0]
                ]
                y_0_ground_truth[level] = (features * t).sum(dim=0)

            for level in range(level_size):
                t_ = data_covs_1[level].flatten().unsqueeze(1)
                features_ = feature_weights[
                    level * t.shape[0] : (level + 1) * t.shape[0]
                ]
                y_1_ground_truth[level] = (features_ * t_).sum(dim=0)
            all_matrices.append((y_0_ground_truth.numpy(), y_1_ground_truth.numpy()))

        return all_matrices

    def _get_count_restrictions_matrix(
        self,
        feature_weights: torch.Tensor,
        counts_matrix: torch.Tensor,
        treatment_level: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        print("Shape of Feature weights: ", feature_weights.shape)
        _, features = feature_weights.shape
        level_size = len(treatment_level)
        print("feature weights: ", feature_weights)
        y_0_ground_truth = torch.zeros(level_size, features)
        y_1_ground_truth = torch.zeros(level_size, features)
        
        print("y_0_ground_truth: ", y_0_ground_truth.shape)
        print("y_1_ground_truth: ", y_1_ground_truth.shape)

        data_count_0 = counts_matrix[0]
        data_count_1 = counts_matrix[1]
        
        print("data_count_0 shape: ", data_count_0.shape)
        print("data_count_1 shape: ", data_count_1.shape)
        print("data_count_0: ", data_count_0)
        print("data_count_1: ", data_count_1)

        print("treatement level: ", treatment_level)
        for level in range(level_size):
            print("level: ", level)
            t = data_count_0[level].flatten().unsqueeze(1)
            print("t: ", t)
            features = feature_weights[level * t.shape[0] : (level + 1) * t.shape[0]]
            print("features: ", features)
            y_0_ground_truth[level] = (features * t).sum(dim=0)

        for level in range(level_size):
            t_ = data_count_1[level].flatten().unsqueeze(1)
            print("t: ", t_)
            features_ = feature_weights[level * t.shape[0] : (level + 1) * t.shape[0]]
            print("features: ", features)
            y_1_ground_truth[level] = (features_ * t_).sum(dim=0)

        print("y_0_ground_truth: ", y_0_ground_truth.numpy())
        print("y_1_ground_truth: ", y_1_ground_truth.numpy())
        return y_0_ground_truth.numpy(), y_1_ground_truth.numpy()

    def build_restriction_matrices(self, feature_weights, strata_estimands):
        restriction_matrices = {
            "count": self._get_count_restrictions_matrix(
                feature_weights,
                strata_estimands["count"],
                self.dataset.levels_colinear[0],
            ),
            "count_minus": self._get_count_restrictions_matrix(
                feature_weights,
                strata_estimands["count"],
                self.dataset.levels_colinear[0],
            ),
            "count_plus": self._get_count_restrictions_matrix(
                feature_weights,
                strata_estimands["count"],
                self.dataset.levels_colinear[0],
            )
            + self._get_count_restrictions_matrix(
                feature_weights,
                strata_estimands["count_plus"],
                self.dataset.levels_colinear[0],
            ),
        }

        if self.n_cov_pairs:
            restriction_matrices["cov"] = self._get_cov_restrictions_matrix(
                feature_weights,
                strata_estimands["cov"],
                self.dataset.levels_colinear[0],
            )

        self.restriction_matrices = restriction_matrices

    def get_cvxpy_restrictions(
        self, cvxpy_weights, feature_weights, ratio, q, n_sample, slack, rho=None
    ):
        """
        Builds constraints in the form needed for the cvxpy package 
        which is the optimization.
        """
        if self.restriction_values is None:
            raise ValueError("Restriction values not set. Run build_restriction_values")
        if self.restriction_matrices is None:
            raise ValueError(
                "Restriction matrices not set. Run build_restriction_matrices"
            )
        dataset_size = self.dataset.population_df_colinear.shape[0]

        print("CVPXY RESTRICTIONS")
        print("self.restriction_matrices[count][0]", self.restriction_matrices["count"][0])
        print("self.restriction_values[count][0]", self.restriction_values["count"][0])
        print("self.restriction_matrices[count][1]", self.restriction_matrices["count"][1])
        print("self.restriction_values[count][1]", self.restriction_values["count"][1])
        print("cvxpy_weights shape", cvxpy_weights.shape)
        print("n_samples: ", n_sample)
        print("dataset_size: ", dataset_size)
        print("first restriction amount: ", n_sample / dataset_size)
        print("feature_weights.shape: ", feature_weights.shape)
        # restrictions = [feature_weights @ cvxpy_weights >= n_sample / dataset_size] # DGG
        restrictions = [feature_weights @ cvxpy_weights >= 0]
        # if self.restriction_type == "count":
        #     restrictions += [ 
        #         self.restriction_matrices["count"][0] @ cvxpy_weights
        #         == self.restriction_values["count"][0],
        #         self.restriction_matrices["count"][1] @ cvxpy_weights
        #         == self.restriction_values["count"][1],
        #     ]
        # epsilon = 1e-2  # Tolerance is needed for the simple case because off the mismatch in conditions
        # if self.restriction_type == "count":
        #     restrictions += [
        #         self.restriction_matrices["count"][0] @ cvxpy_weights
        #         >= self.restriction_values["count"][0] - epsilon,
        #         self.restriction_matrices["count"][1] @ cvxpy_weights
        #         >= self.restriction_values["count"][1] - epsilon,
        #     ]
        if self.restriction_type == "count":
            restrictions += [
                self.restriction_matrices["count"][0] @ cvxpy_weights + slack[0]
                == self.restriction_values["count"][0],
                self.restriction_matrices["count"][1] @ cvxpy_weights + slack[1]
                == self.restriction_values["count"][1],
            ]
        elif self.restriction_type == "count_minus":
            restrictions += [
                self.restriction_matrices["count_minus"][0] @ cvxpy_weights
                == self.restriction_values["count_minus"][0]
            ]
        elif self.restriction_type == "count_plus":
            restrictions += [
                self.restriction_matrices["count_plus"][0] @ cvxpy_weights
                == self.restriction_values["count_plus"][0],
                self.restriction_matrices["count_plus"][1] @ cvxpy_weights
                == self.restriction_values["count_plus"][1],
                self.restriction_matrices["count_plus"][3] @ cvxpy_weights
                == self.restriction_values["count_plus"][3],
            ]
        elif self.restriction_type == "cov_positive":
            for cov_pair in self.restriction_matrices["cov"]:
                restrictions += [
                    cov_pair[0] @ cvxpy_weights >= 0,
                    cov_pair[1] @ cvxpy_weights >= 0,
                ]
        elif self.restriction_type == "cov_negative":
            for cov_pair in self.restriction_matrices["cov"]:
                restrictions += [
                    cov_pair[0] @ cvxpy_weights <= 0,
                    cov_pair[1] @ cvxpy_weights <= 0,
                ]
        elif self.restriction_type == "cov_strict":
            for cov_pair, restriction_pair in zip(
                self.restriction_matrices["cov"], self.cons_values["cov"]
            ):
                restrictions += [
                    cov_pair[0] @ cvxpy_weights >= restriction_pair[0],
                    cov_pair[1] @ cvxpy_weights >= restriction_pair[1],
                ]
        elif (
            self.restriction_type == "DRO" or self.restriction_type == "DRO_worst_case"
        ):
            chi_square_divergence = cp.multiply((ratio - 1) ** 2, q)
            restrictions += [0.5 * cp.sum(chi_square_divergence) <= rho]
        else:
            raise ValueError(f"Restriction type {self.restriction_type} not supported.")
        return restrictions


class SemiSyntheticRestrictions(Restrictions):
    def __init__(
        self, dataset, restriction_type: str, n_cov_pairs: Optional[int] = None
    ):
        self.restriction_type = restriction_type
        self.n_cov_pairs = n_cov_pairs
        self.dataset = dataset
        self.restriction_values = None
        self.restriction_matrices = None

    def _get_count_restrictions(
        self, data: pd.DataFrame, target: str, treatment_level: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        y00_val_1 = sum((data[target[0]] == 1) & (data[treatment_level[0]] == 1))
        y01_val_1 = sum((data[target[1]] == 1) & (data[treatment_level[0]] == 1))

        y00_val_2 = sum((data[target[0]] == 1) & (data[treatment_level[1]] == 1))
        y01_val_2 = sum((data[target[1]] == 1) & (data[treatment_level[1]] == 1))

        restriction_00 = np.array([y00_val_1, y00_val_2])
        restriction_01 = np.array([y01_val_1, y01_val_2])

        return restriction_00, restriction_01

    def _get_count_restrictions_y(
        self, data: pd.DataFrame, target: str, treatment_level: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        y00_val_1 = sum((data[target] == 0) & (data[treatment_level[0]] == 1))
        y01_val_1 = sum((data[target] == 1) & (data[treatment_level[0]] == 1))

        y00_val_2 = sum((data[target] == 0) & (data[treatment_level[1]] == 1))
        y01_val_2 = sum((data[target] == 1) & (data[treatment_level[1]] == 1))

        restriction_00 = np.array([y00_val_1, y00_val_2])
        restriction_01 = np.array([y01_val_1, y01_val_2])

        return restriction_00, restriction_01

    def _get_cov_restrictions(
        self,
        data: pd.DataFrame,
        target: str,
        treatment_level: List[str],
        all_cov_vars: List[List[str]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_00_val_1_df = data[(data[target[0]] == 1) & (data[treatment_level[0]] == 1)]
        y_01_val_1_df = data[(data[target[1]] == 1) & (data[treatment_level[0]] == 1)]

        y_00_val_2_df = data[(data[target[0]] == 1) & (data[treatment_level[1]] == 1)]
        y_01_val_2_df = data[(data[target[1]] == 1) & (data[treatment_level[1]] == 1)]

        all_cov_restrictions = []
        for cov_vars in all_cov_vars:
            cov_vars_unpacked = [*cov_vars]
            y_00_val_1 = y_00_val_1_df.cov().loc[cov_vars_unpacked]
            y_01_val_1 = y_01_val_1_df.cov().loc[cov_vars_unpacked]
            y_00_val_2 = y_00_val_2_df.cov().loc[cov_vars_unpacked]
            y_01_val_2 = y_01_val_2_df.cov().loc[cov_vars_unpacked]
            # y_00_val_1 = y_00_val_1_df.cov().loc[*cov_vars]
            # y_01_val_1 = y_01_val_1_df.cov().loc[*cov_vars]
            # y_00_val_2 = y_00_val_2_df.cov().loc[*cov_vars]
            # y_01_val_2 = y_01_val_2_df.cov().loc[*cov_vars]

            restriction_00 = np.array([y_00_val_1, y_00_val_2])
            restriction_01 = np.array([y_01_val_1, y_01_val_2])
            all_cov_restrictions.append((restriction_00, restriction_01))
        return all_cov_restrictions

    def _get_cov_pairs(self, n_pairs, treatment_level, mode):
        input_df = self.dataset.population_df_colinear.copy(())
        all_cols = input_df.columns.to_list()
        first_group_prefix = all_cols[0].split("_")[0]

        first_group_cols = input_df.filter(
            regex=f"^{first_group_prefix}_"
        ).columns.to_list()
        excluded_cols = first_group_cols + [self.dataset.target] + treatment_level
        selected_cols = [col for col in all_cols if col not in excluded_cols]
        covs = input_df.cov().loc[selected_cols, first_group_cols].unstack()

        if mode == "positive":
            covs = covs[covs >= 0]
            cov_pairs = covs.sort_values(ascending=False).index.to_list()[:n_pairs]

        elif mode == "negative":
            covs = covs[covs <= 0]
            cov_pairs = covs.sort_values(ascending=True).index.to_list()[:n_pairs]

        elif mode == "strict":
            cov_pairs = covs.sort_values(ascending=False).index.to_list()[:n_pairs]
        return cov_pairs

    def build_restriction_values(self) -> None:
        restriction_values = {
            "count": self._get_count_restrictions(
                data=self.dataset.population_df_colinear,
                target=self.dataset.levels_colinear[-2],
                treatment_level=self.dataset.levels_colinear[-1],
            ),
            "count_minus": self._get_count_restrictions(
                data=self.dataset.population_df_colinear,
                target=self.dataset.levels_colinear[-2],
                treatment_level=self.dataset.levels_colinear[-1],
            ),
            "count_plus": self._get_count_restrictions(
                data=self.dataset.population_df_colinear,
                target=self.dataset.levels_colinear[-2],
                treatment_level=self.dataset.levels_colinear[-1],
            )
            + self._get_count_restrictions_y(
                data=self.dataset.population_df_colinear,
                target=self.dataset.target,
                treatment_level=self.dataset.levels_colinear[0],
            ),
        }
        

        if self.n_cov_pairs and self.restriction_type.startswith("cov"):
            all_cov_vars = self._get_cov_pairs(
                n_pairs=self.n_cov_pairs,
                treatment_level=self.dataset.levels_colinear[0],
                mode=self.restriction_type.split("_")[-1],
            )

            restriction_values["cov"] = self._get_cov_restrictions(
                data=self.dataset.population_df_colinear,
                target=self.dataset.levels_colinear[-2],
                treatment_level=self.dataset.levels_colinear[-1],
                all_cov_vars=all_cov_vars,
            )

            all_feature_means = []
            for cov_vars in all_cov_vars:
                feature_means = {
                    cov_vars[0]: self.dataset.population_df_colinear[
                        cov_vars[0]
                    ].mean(),
                    cov_vars[1]: self.dataset.population_df_colinear[
                        cov_vars[1]
                    ].mean(),
                }
                all_feature_means.append(feature_means)
            self.all_feature_means = all_feature_means

        self.restriction_values = restriction_values

    def _get_cov_restrictions_matrix(
        self,
        feature_weights: torch.Tensor,
        all_covs: torch.Tensor,
        treatment_level: list[str],
    ):
        _, features = feature_weights.shape
        level_size = len(treatment_level)

        y_0_ground_truth = torch.zeros(level_size, features)
        y_1_ground_truth = torch.zeros(level_size, features)

        all_matrices = []
        for covs in all_covs:
            data_covs_0 = covs[0]
            data_covs_1 = covs[1]

            for level in range(level_size):
                t = data_covs_0[level].flatten().unsqueeze(1)
                features = feature_weights[
                    level * t.shape[0] : (level + 1) * t.shape[0]
                ]
                y_0_ground_truth[level] = (features * t).sum(dim=0)

            for level in range(level_size):
                t_ = data_covs_1[level].flatten().unsqueeze(1)
                features_ = feature_weights[
                    level * t.shape[0] : (level + 1) * t.shape[0]
                ]
                y_1_ground_truth[level] = (features_ * t_).sum(dim=0)
            all_matrices.append((y_0_ground_truth.numpy(), y_1_ground_truth.numpy()))

        return all_matrices

    def _get_count_restrictions_matrix(
        self,
        feature_weights: torch.Tensor,
        counts_matrix: torch.Tensor,
        treatment_level: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        _, features_n = feature_weights.shape
        level_size = len(treatment_level)

        data_count_0 = counts_matrix.select(-2, 0)
        data_count_1 = counts_matrix.select(-2, 1)

        features_0 = []
        features_1 = []

        for i in range(feature_weights.shape[0]):
            if i % 4 == 0 or i % 4 == 1:
                features_0.append(feature_weights[i])
            else:
                features_1.append(feature_weights[i])

        features_0 = torch.stack(features_0)
        features_1 = torch.stack(features_1)

        y_0_treatment = torch.zeros(level_size, features_n)
        y_1_treatment = torch.zeros(level_size, features_n)

        for level in range(level_size):
            t = data_count_0.select(-1, level)
            t = t.sum(axis=0).flatten().unsqueeze(1)
            features = features_0[level::2]
            y_0_treatment[level] = (features * t).sum(dim=0)

        for level in range(level_size):
            t_ = data_count_1.select(-1, level)
            t_ = t_.sum(axis=0).flatten().unsqueeze(1)
            features_ = features_1[level::2]
            y_1_treatment[level] = (features_ * t_).sum(dim=0)

        return y_0_treatment.numpy(), y_1_treatment.numpy()

    def _get_outcome_count_restrictions_matrix(
        self,
        feature_weights: torch.Tensor,
        counts: torch.Tensor,
        treatment_level: list[str],
    ):
        _, features_n = feature_weights.shape
        level_size = len(treatment_level)

        y_0_treatment = torch.zeros(level_size, features_n)
        y_1_treatment = torch.zeros(level_size, features_n)

        data_count_0 = counts[0]
        data_count_1 = counts[1]

        for level in range(2):
            t = data_count_0[level].flatten().unsqueeze(1)
            features = feature_weights[level * t.shape[0] : (level + 1) * t.shape[0]]
            y_0_treatment[level] = (features * t).sum(dim=0)

        for level in range(level_size):
            t_ = data_count_1[level].flatten().unsqueeze(1)
            features_ = feature_weights[level * t_.shape[0] : (level + 1) * t_.shape[0]]
            y_1_treatment[level] = (features_ * t_).sum(dim=0)

        return y_0_treatment.numpy(), y_1_treatment.numpy()

    def build_restriction_matrices(self, feature_weights, strata_estimands):
        restriction_matrices = {
            "count": self._get_count_restrictions_matrix(
                feature_weights,
                strata_estimands["count"],
                self.dataset.levels_colinear[-1],
            ),
            "count_minus": self._get_count_restrictions_matrix(
                feature_weights,
                strata_estimands["count"],
                self.dataset.levels_colinear[-1],
            ),
            "count_plus": self._get_count_restrictions_matrix(
                feature_weights,
                strata_estimands["count"],
                self.dataset.levels_colinear[-1],
            )
            + self._get_outcome_count_restrictions_matrix(
                feature_weights,
                strata_estimands["count_plus"],
                self.dataset.levels_colinear[0],
            ),
        }

        if self.n_cov_pairs and self.restriction_type.startswith("cov"):
            restriction_matrices["cov"] = self._get_cov_restrictions_matrix(
                feature_weights,
                strata_estimands["cov"],
                self.dataset.levels_colinear[0],
            )

        self.restriction_matrices = restriction_matrices

    def get_cvxpy_restrictions(
        self, cvxpy_weights, feature_weights, ratio, q, n_sample, rho=None
    ):
        if self.restriction_values is None:
            raise ValueError("Restriction values not set. Run build_restriction_values")
        if self.restriction_matrices is None:
            raise ValueError(
                "Restriction matrices not set. Run build_restriction_matrices"
            )

        dataset_size = self.dataset.population_df_colinear.shape[0]

        restrictions = [feature_weights @ cvxpy_weights >= n_sample / dataset_size]
        if self.restriction_type == "count":
            restrictions += [
                self.restriction_matrices["count"][0] @ cvxpy_weights
                == self.restriction_values["count"][0],
                self.restriction_matrices["count"][1] @ cvxpy_weights
                == self.restriction_values["count"][1],
            ]
        elif self.restriction_type == "count_minus":
            restrictions += [
                self.restriction_matrices["count_minus"][0] @ cvxpy_weights
                == self.restriction_values["count_minus"][0]
            ]
        elif self.restriction_type == "count_plus":
            restrictions += [
                self.restriction_matrices["count_plus"][0] @ cvxpy_weights
                == self.restriction_values["count_plus"][0],
                self.restriction_matrices["count_plus"][1] @ cvxpy_weights
                == self.restriction_values["count_plus"][1],
                self.restriction_matrices["count_plus"][2] @ cvxpy_weights
                == self.restriction_values["count_plus"][2],
                self.restriction_matrices["count_plus"][3] @ cvxpy_weights
                == self.restriction_values["count_plus"][3],
            ]
        elif self.restriction_type == "cov_positive":
            for cov_pair in self.restriction_matrices["cov"]:
                restrictions += [
                    cov_pair[0] @ cvxpy_weights >= 0,
                    cov_pair[1] @ cvxpy_weights >= 0,
                ]
        elif self.restriction_type == "cov_negative":
            for cov_pair in self.restriction_matrices["cov"]:
                restrictions += [
                    cov_pair[0] @ cvxpy_weights <= 0,
                    cov_pair[1] @ cvxpy_weights <= 0,
                ]
        elif self.restriction_type == "cov_strict":
            for cov_pair, restriction_pair in zip(
                self.restriction_matrices["cov"], self.restriction_values["cov"]
            ):
                restrictions += [
                    cov_pair[0] @ cvxpy_weights >= restriction_pair[0],
                    cov_pair[1] @ cvxpy_weights >= restriction_pair[1],
                ]
        elif (
            self.restriction_type == "DRO" or self.restriction_type == "DRO_worst_case"
        ):
            chi_square_divergence = cp.multiply((ratio - 1) ** 2, q)
            restrictions += [0.5 * cp.sum(chi_square_divergence) <= rho]
        else:
            raise ValueError(f"Restriction type {self.restriction_type} not supported.")
        return restrictions
