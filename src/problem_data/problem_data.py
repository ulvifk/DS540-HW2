import pandas as pd
from .pre_processor import PreProcessor


class ProblemData:
    df: pd.DataFrame
    df_X: pd.DataFrame
    df_Y: pd.DataFrame

    def __init__(self, path: str):
        self.df = pd.read_csv(path, index_col='customerID')
        self.df = PreProcessor(self.df).df
        self.df_X = self.df.drop('Churn', axis=1)
        self.df_Y = self.df[['Churn']]
