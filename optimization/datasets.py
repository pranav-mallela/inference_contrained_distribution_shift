from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment, ACSIncome
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Dataset:
    population_df: pd.DataFrame
    sample_df: pd.DataFrame
    levels: List[List[str]]
    target: str
    alternate_outcome: str
    empirical_conditional_mean: float
    true_conditional_mean: float
    population_df_colinear: pd.DataFrame = None
    sample_df_colinear: pd.DataFrame = None
    levels_colinear: List[List[str]] = None


class DatasetLoader(ABC):
    @abstractmethod
    def load(self) -> Dataset:
        pass

class SimulationLoaderOurs(DatasetLoader):
    def __init__(self, dataset_size: int, correlation_coeff: float, rng: np.random.Generator) -> None:
        self.dataset_size = dataset_size
        self.correlation_coeff = correlation_coeff
        self.rng = rng

    def _simulate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulates the dataset with hospitalization and race probabilities.
        
        Returns:
            Tuple: A tuple containing the population dataset and the observed sample dataset.
        """
        # Simulate population dataset
        P = pd.DataFrame({
            'race': self.rng.choice(['majority', 'minority'], size=self.dataset_size, p=[0.7, 0.3]),
            'hospitalized': self.rng.choice([0, 1], size=self.dataset_size, p=[0.6, 0.4]),
            'other_outcome': self.rng.choice([0, 1], size=self.dataset_size, p=[0.5, 0.5]) #this to match the dataset format but we are not using it
        })

        # Define observed probabilities for sampling
        observed_probs = {
            ('majority', 1): 0.7,
            ('minority', 1): 0.3
        }

        # Calculate weights for biased sampling
        def calculate_weight(row):
            if row['hospitalized'] == 1:
                return observed_probs.get((row['race'], row['hospitalized']), 1.0)
            return 1.0

        P['weight'] = P.apply(calculate_weight, axis=1)

        # Create observed dataset Q with biased sampling
        Q = P.sample(n=self.dataset_size, weights=P['weight'], replace=True).drop(columns=['weight'])
        
        return P, Q

    def _create_dataframe(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the dataset into a formatted dataframe.
        
        Args:
            dataset (pd.DataFrame): The dataset to format.

        Returns:
            pd.DataFrame: A formatted dataframe with dummies.
        """
        df = pd.DataFrame()
        df[['majority', 'minority']] = pd.get_dummies(dataset['race'])
        df['hospitalized'] = dataset['hospitalized']
        df['other_outcome'] = dataset['other_outcome']
        return df
    
    def _get_ate_conditional_mean(self, A: np.ndarray, y: np.ndarray) -> float:
        """Computes the ate using inverse propensity weighting.

        Args:
            A (np.ndarray): vector of observed outcomes
            propensity_scores (_type_): Vector of propensity scores
            y (np.ndarray): response variable

        Returns
            ate: Average treatment effect.
        """
        conditional_mean = (y * A).sum()
        return conditional_mean / A.sum()

    def load(self) -> Dataset:
        """
        Loads the population and sample datasets, computes statistics, and returns them as a Dataset object.
        
        Returns:
            Dataset: A Dataset object containing population and sample data.
        """
        # Generate the population and observed sample datasets
        P, Q = self._simulate_dataset()

        # Create formatted dataframes
        population_df = self._create_dataframe(P)
        sample_df = self._create_dataframe(Q)
        
        print("population df: ", population_df)
        print("sample df: ", sample_df)
        
        y = P['hospitalized'].values
        y_raw = Q['hospitalized'].values
        X = population_df[['majority', 'minority']].values  
        X_raw = sample_df[['majority', 'minority']].values 
        
        levels = [
            ['majority', 'minority'],
        ]

        empirical_conditional_mean = self._get_ate_conditional_mean(X[:, -1], y)
        true_conditional_mean = self._get_ate_conditional_mean(X_raw[:, -1], y_raw)

        # Create and return Dataset object
        dataset = Dataset(
            population_df=population_df,
            sample_df=sample_df,
            population_df_colinear=population_df,
            sample_df_colinear=sample_df,
            levels=levels,
            levels_colinear=levels,
            target='hospitalized',
            alternate_outcome='other_outcome',
            empirical_conditional_mean=empirical_conditional_mean,
            true_conditional_mean=true_conditional_mean,
        )
        return dataset

    
class SimulationLoader(DatasetLoader):
    def __init__(
        self, dataset_size: int, correlation_coeff: float, rng: np.random.Generator
    ) -> None:
        self.dataset_size = dataset_size
        self.correlation_coeff = correlation_coeff
        self.rng = rng

    def _simulate_dataset(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulates a dataset with multiple outcomes.

        The following function simulates a data generating process with multiple
        outcomes. Moreover, it also generates a sample of observations from the
        original dataset.

        Args:
            dataset_size (int): The dataset size.
            rng (np.random.Generator): The random number generator.
            correlation_coeff (float, optional): The correlation coefficient
            to determine influence of X_1 and X_2 on indicator used to sample
            the biased dataset.

        Returns:
            Tuple: the sample from the generated dataset.
        """
        X = self.rng.choice(a=[0, 1, 2], size=self.dataset_size, p=[0.5, 0.3, 0.2])
        X_2 = self.rng.binomial(size=self.dataset_size, n=1, p=0.4)

        pi_A = expit(X_2 - X)
        A = 1 * (pi_A > self.rng.uniform(size=self.dataset_size))
        mu = expit(2 * A - X + X_2)
        y = 1 * (mu > self.rng.uniform(size=self.dataset_size))

        mu2 = expit((X + X_2) / 2 - A)
        y2 = 1 * (mu2 > self.rng.uniform(size=self.dataset_size))
        
        # Single Variable - Indirect Dependency: Jul23-2020, Jul23-0957
        # obs = expit(A) > self.rng.uniform(size=self.dataset_size)
        
        # Single Variable: Jul23-0956, Jul23-0957
        # obs = expit(X) > self.rng.uniform(size=self.dataset_size)
        
        # Changed Function 2: Jul22-2219, Jul22-2220
        # obs = expit(A + y2) > self.rng.uniform(size=self.dataset_size)

        # Remove Direct Dependency: Jul22-2006, Jul22-2008; Jul23-0924, Jul23-0925
        # obs = expit(A - y2) > self.rng.uniform(size=self.dataset_size)

        # Inverse Dependency: Jul22-1858
        # obs = expit(X_2 - X) > self.rng.uniform(size=self.dataset_size)

        # Changed Function: Jul22-1939, Jul22-1941
        # obs = expit(X + X_2) > self.rng.uniform(size=self.dataset_size)

        # Original: Jul22-1921, Jul22-1925; Jul24-0846, Jul24-0847
        obs = expit(X - X_2) > self.rng.uniform(size=self.dataset_size)

        X_total = np.stack((X, X_2), axis=-1)

        return X_total, A, y, obs, y2

    def _create_dataframe(self, X: np.ndarray, A: np.ndarray) -> pd.DataFrame:
        """Creates a dataframe with the data.

        The data creates dummie variables from categorical variables passed as
        parameters as well as the treatment variable.
        """
        df = pd.DataFrame()
        df[["little", "moderate", "quite rich"]] = pd.get_dummies(X[:, 0])
        df[["White", "Non White"]] = pd.get_dummies(X[:, 1])
        df[["female", "male"]] = pd.get_dummies(A)
        return df

    def _get_ate_conditional_mean(self, A, y):
        """Computes the ate using inverse propensity weighting.

        Args:
            A (_type_): vector of observed outcomes
            propensity_scores (_type_): Vector of propensity scores
            y (_type_): response variable

        Returns
            ate: Average treatment effect.
        """
        conditional_mean = (y * A).sum()
        return conditional_mean / A.sum()

    def load(self) -> Dataset:
        levels = [
            ["female", "male"],
            ["little", "moderate", "quite rich"],
            ["White", "Non White"],
        ]

        levels_colinear = levels

        X_raw, A_raw, y_raw, obs, y_2_raw = self._simulate_dataset()

        X = X_raw[obs]
        A = A_raw[obs]
        y = y_raw[obs]
        y_2 = y_2_raw[obs]

        empirical_conditional_mean = self._get_ate_conditional_mean(X[:, -1], y)
        true_conditional_mean = self._get_ate_conditional_mean(X_raw[:, -1], y_raw)

        sample_df = self._create_dataframe(X, A)
        sample_df["Creditability"] = y
        sample_df["other_outcome"] = y_2

        population_df = self._create_dataframe(X_raw, A_raw)
        print("population_data: ", population_df)
        population_df["Creditability"] = y_raw
        population_df["other_outcome"] = y_2_raw

        print("population_data: ", population_df)
        population_df_colinear = population_df.copy()
        sample_df_colinear = sample_df.copy()

        print("Population Size: ", len(population_df))
        print("Sample size: ", len(sample_df))
        dataset = Dataset(
            population_df=population_df,
            sample_df=sample_df,
            population_df_colinear=population_df_colinear,
            sample_df_colinear=sample_df_colinear,
            levels=levels,
            levels_colinear=levels_colinear,
            target="Creditability",
            alternate_outcome="other_outcome",
            empirical_conditional_mean=empirical_conditional_mean,
            true_conditional_mean=true_conditional_mean,
        )
        return dataset


class FolktablesLoader(DatasetLoader):
    def __init__(
        self,
        feature_names: List[str],
        states: List[str],
        rng: np.random.Generator,
        survey_year: str = "2018",
        horizon: str = "1-Year",
        survey: str = "person",
        alternate_outcome: str = "DIS_1",
        conditioning_features: List[str] = None,
        size=None,
        buckets: dict = None,
        correlation_coeff: float = 1,
    ) -> None:
        self.feature_names = feature_names
        self.states = states
        self.rng = rng
        self.survey_year = survey_year
        self.horizon = horizon
        self.survey = survey
        self.alternate_outcome = alternate_outcome
        self.size = size
        self.correlation_coeff = correlation_coeff
        if not conditioning_features:
            self.conditioning_features = feature_names[:2]
        if buckets:
            self.buckets = buckets
        else:
            self.buckets = {
                "SCHL": {
                    (0, 0): 0,  # less-than-3
                    (1, 1): 1,  # no-school
                    (2, 3): 2,  # preschool
                    (4, 8): 3,  # elementary
                    (9, 16): 4,  # high-school
                    (17, 21): 5,  # college
                    (22, 24): 6,  # graduate
                }
            }

    def _download_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_source = ACSDataSource(
            survey_year=self.survey_year, horizon=self.horizon, survey=self.survey
        )

        acs_data = data_source.get_data(states=self.states, download=True)
        ACSIncome._preprocess = ACSEmployment._preprocess

        income = ACSIncome.df_to_numpy(acs_data)[1]
        X, y, group = ACSEmployment.df_to_numpy(acs_data)

        A = 1 * (group == 1)

        X = X.astype(int)
        y = y.astype(int)

        def transform_values(value):
            if value < 20:
                return 1
            elif 20 <= value <= 24:
                return 2
            else:
                return None

        mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}

        # last feature is the group
        df = pd.DataFrame(X, columns=ACSEmployment.features)
        # Apply the mapping to the column
        df["PINCP"] = income.astype(int)

        df = df[self.feature_names]
        df["MIL"] = df["MIL"].map(mapping)
        X_selected = df.to_numpy()

        X_normed = MinMaxScaler().fit_transform(X_selected)
        obs = expit(
            -X_normed[:, 0] - self.correlation_coeff * X_normed[:, 1]
        ) > self.rng.uniform(size=X.shape[0])
        size = X.shape[0] if not self.size else self.size
        return X_selected[:size], A[:size], y[:size], obs[:size]

    def _group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature, feature_buckets in self.buckets.items():
            bins = [-1] + [edges[1] for edges in feature_buckets.keys()]
            df[feature] = pd.cut(
                df[feature], bins=bins, labels=list(feature_buckets.values())
            )
        return df

    def _get_ate_conditional_mean(self, A: np.ndarray, y: np.ndarray) -> float:
        """Computes the ate using inverse propensity weighting.

        Args:
            A (np.ndarray): vector of observed outcomes
            propensity_scores (_type_): Vector of propensity scores
            y (np.ndarray): response variable

        Returns
            ate: Average treatment effect.
        """
        conditional_mean = (y * A).sum()
        return conditional_mean / A.sum()

    def _create_dataframe(
        self, X: np.ndarray, A: np.ndarray, feature_names: List[str], full=True
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        data = pd.DataFrame()
        levels = []

        for column_idx in range(X.shape[1]):
            features = pd.get_dummies(X[:, column_idx])
            strata_number = features.shape[1]
            names = [
                str(feature_names[column_idx]) + "_" + str(j)
                for j in range(int(strata_number))
            ]
            if full:
                data[names] = 0
                data[names] = features
                levels.append(names)
            else:
                data[names[:-1]] = 0
                data[names[:-1]] = features[features.columns[:-1]]
                levels.append(names[:-1])
        if full:
            data[["white", "non-white"]] = pd.get_dummies(A)
        else:
            data["white"] = pd.get_dummies(A)[0]
        return data, levels

    def load(self) -> Dataset:
        X, A, y, obs = self._download_data()

        X_sample = X[obs]
        y_sample = y[obs]

        empirical_conditional_mean = self._get_ate_conditional_mean(
            1 - X_sample[:, -1], y_sample
        )

        true_conditional_mean = self._get_ate_conditional_mean(1 - X[:, -1], y)

        population_df, levels = self._create_dataframe(
            X, A, self.feature_names, full=False
        )
        levels = [["white"]] + levels
        population_df["Creditability"] = y

        sample_df = population_df[obs].copy()
        sample_df["Creditability"] = y_sample

        population_df_colinear, levels_colinear = self._create_dataframe(
            X, A, self.feature_names, full=True
        )
        levels_colinear = [["white", "non-white"]] + levels_colinear
        population_df_colinear["Creditability"] = y

        sample_df_colinear = population_df_colinear[obs].copy()
        sample_df_colinear["Creditability"] = y_sample

        dataset = Dataset(
            population_df=population_df,
            sample_df=sample_df,
            population_df_colinear=population_df_colinear,
            sample_df_colinear=sample_df_colinear,
            levels=levels,
            levels_colinear=levels_colinear,
            target="Creditability",
            alternate_outcome=self.alternate_outcome,
            empirical_conditional_mean=empirical_conditional_mean,
            true_conditional_mean=true_conditional_mean,
        )
        return dataset

