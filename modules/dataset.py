import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

class StoreDataset:

    # url to datasets
    __url_definitivo = "/content/drive/MyDrive/university/msc/pg_businesscase/data/definitivo.csv"

    # define the demographic columns
    __demographic = ['maschi', 'femmine', 'celibi/nubili', 'coniugati', 'divorziati', 'eta_minore_5', 'eta_5/9', 
                    'eta_10/14', 'eta_15/19', 'eta_20/24', 'eta_25/29', 'eta_30/34', 'eta_35/39', 'eta_40/44', 
                    'eta_45/49', 'eta_50/54', 'eta_55/59', 'eta_60/64', 'eta_65/69', 'eta_70/74', 'eta_maggiore_74', 
                    'con_laurea', 'con_diploma', 'con_media', 'con_elementare', 'occupati', 'disoccupati', 'studenti','single']
    
    __replacements = {   
                    'indice_d': {
                                    'D1': 1,
                                    'D2': 2,
                                    'D3': 3,
                                    'D4': 4,
                                    'D5': 5
                                }, 
                    'indice_p': {
                                    'P1': 1,
                                    'P2': 2,
                                    'P3': 3,
                                    'P4': 4,
                                    'P5': 5
                                }, 
                    'indice_t': {
                                    'T1': 1,
                                    'T2': 2,
                                    'T3': 3,
                                    'T4': 4,
                                    'T5': 5
                                }, 
                    'indice_sintesi': {
                                    'S1': 1,
                                    'S2': 2,
                                    'S3': 3,
                                    'S4': 4,
                                    'S5': 5
                                }
                    }
    
    
    def __init__(self, url: str=__url_definitivo):
        """Class constructor"""
        self.initial = pd.read_csv(url)
        self.initial = self.initial.drop(columns=['Unnamed: 0'])
        
        self.initial = self.initial.replace(StoreDataset.__replacements)

        self.train_set = self.initial[self.initial['vendite'] > 0.0]
        self.test_set = self.initial[self.initial['vendite'] == 0.0]


    def get_datasets(self, threshold: float=0.1) -> (pd.core.frame.DataFrame, pd.core.frame.DataFrame, pd.core.frame.DataFrame):
        """ Returns the train and test datasets.
        Parameters
        ----------
            threshold: float (default=0.1)
                threshold used to remove the columns which has a correlation 
                (in absolute value) with the 'vendite' column less than that value
        
        Returns
        -------
            X_train: pandas.core.frame.DataFrame
                dataset to be used as train input
            Y_train: pandas.core.frame.DataFrame
                dataset to be used as target input
            test: pandas.core.frame.DataFrame
                dataset to be used as test (target must be predicted)
        """
        Y_train = self.train_set['vendite']

        features_to_use_name = self.correlated_features(dataframe=self.train_set, 
                                                        reference_feature='vendite', 
                                                        threshold=threshold)

        X_train = self.train_set[features_to_use_name]
        X_train = X_train.drop(columns=['vendite'])

        test = self.test_set[features_to_use_name]
        test = test.drop(columns=['vendite'])

        return (X_train, Y_train, test)

    
    def update_with_results(self, results: np.ndarray, sort_by_sales: bool=True):
        self.final = self.initial.copy()
        self.final.loc[self.final['vendite'] == 0, 'vendite'] = results
        if sort_by_sales:
            self.final = self.final.sort_values(by='vendite', ascending=False)
        self.final.reset_index(inplace=True, drop=True)

    
    def update_with_clusters(self, labels: np.ndarray):
        self.clusterized = self.final.copy()
        self.clusterized['cluster'] = labels
        self.clusterized.reset_index(inplace=True, drop=True)

    
    def correlated_features(self, dataframe, reference_feature, threshold):
        corr_matrix = dataframe.corr()[reference_feature]
        mask = abs(corr_matrix) > threshold
        features_to_use = corr_matrix[mask]
        return features_to_use.index


    def plot_clusters_ll(self):
        if hasattr(self, 'clusterized'):
            plot_dims=(12,8)
            plt.figure(figsize=plot_dims)
            sns.scatterplot(x=self.clusterized.longitude, y=self.clusterized.latitude, 
                            hue = self.clusterized.cluster, palette=cm.tab10)
        else:
            raise ValueError('Create clusters first. Call method `update_with_clusters()`')


def plot_correlation_matrix(dataframe: pd.core.frame.DataFrame):
    """ Plots the correlation matrix of the given dataframe.
    Parameters
    ---------- 
        dataframe: pandas.core.frame.DataFrame
            dataframe of which to plot the correlation matrix
    """
    plt.figure(figsize=(50,50))
    plt.title('Correlation of Features')
    sns.heatmap(dataframe.corr(), linewidths=0.2, vmax=1.0, vmin=-1.0,
                square=True, cmap=cm.RdBu, linecolor='white', annot=True, fmt='.2f')


def prepare_for_clustering(dataframe: pd.core.frame.DataFrame, cols_to_drop: list=None, cols_to_keep: list=None) -> pd.core.frame.DataFrame:
    
    if (cols_to_drop is None and cols_to_keep is None) or (cols_to_drop is not None and cols_to_keep is not None):
        raise ValueError("One between cols_to_drop and cols_to_keep must be defined.")

    if cols_to_drop is not None:
        return dataframe.drop(columns=cols_to_drop)

    elif cols_to_keep is not None:
        return dataframe[cols_to_keep]


def object_to_float(dataframe: pd.core.frame.DataFrame, cols: list) -> pd.core.frame.DataFrame:
    for col in cols:
        dataframe[col] = dataframe[col].map(lambda x: float(x.replace(',','.')) if isinstance(x, str) else x)
    
    return dataframe