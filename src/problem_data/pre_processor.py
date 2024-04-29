import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class PreProcessor:
    df: pd.DataFrame
    cols_to_one_hot: list[str]
    cols_to_label_encode: list[str]


    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._cast_to_float()
        self.df.dropna(inplace=True)

        self.cols_to_one_hot = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                           "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
        non_numeric_cols = df.select_dtypes(include='object').columns
        self.cols_to_label_encode = [col for col in non_numeric_cols if col not in self.cols_to_one_hot]

        self._one_hot_encode()
        self._label_encode()


    def _cast_to_float(self):
        float_cols = ["MonthlyCharges", "TotalCharges"]
        self.df[float_cols] = self.df[float_cols].apply(pd.to_numeric, errors='coerce')

    def _one_hot_encode(self):
        for col in self.cols_to_one_hot:
            onehot_encoder = OneHotEncoder()
            col_to_encoded = self.df[col].values.reshape(-1, 1)
            self.df.drop(col, axis=1, inplace=True)

            onehot_encoded = onehot_encoder.fit_transform(col_to_encoded).toarray()
            categories = onehot_encoder.categories_[0]
            col_names = [str(col) + "_" + category for category in categories]

            self.df = pd.concat([self.df, pd.DataFrame(onehot_encoded, columns=col_names, index=self.df.index)], axis=1)

    def _label_encode(self):
        for col in self.cols_to_label_encode:
            label_encoder = LabelEncoder()
            self.df[col] = label_encoder.fit_transform(self.df[col])


