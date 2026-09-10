import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import SEED
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
from shutil import rmtree
import joblib


class SelectiveDropper(TransformerMixin, BaseEstimator):
    """Drop specified NoExposure columns from one-hot encoded features."""

    _sklearn_output_config = {"transform": "default"}  # Add this class attribute

    def __init__(self, cols_with_noexposure):
        self.cols_with_noexposure = cols_with_noexposure
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            return X

        cols_to_drop = [
            col
            for col in X.columns
            if any(
                f"cat__{orig}_NoExposure" in col for orig in self.cols_with_noexposure
            )
        ]

        result = X.drop(columns=cols_to_drop)
        self.feature_names_out_ = result.columns.tolist()

        return result

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
            if input_features is None:
                raise ValueError(
                    "Transformer not fitted and input_features not provided."
                )

            cols_to_drop = [
                col
                for col in input_features
                if any(
                    f"cat__{orig}_NoExposure" in col
                    for orig in self.cols_with_noexposure
                )
            ]
            self.feature_names_out_ = [
                col for col in input_features if col not in cols_to_drop
            ]

        return np.array(self.feature_names_out_)


def remove_prefix(df):
    X = df.copy()
    X.columns = X.columns.str.replace(r"^\w+__", "", regex=True)
    return X


def transform_export_data(X, y, preprocessor, nomo, data_path=None, pipeline_path=None):
    print("\t Splitting data...")
    ##Get train set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    print("\t Fitting preprocessor...")
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out()

    print("\t Transforming train, val, and test sets...")
    X_train_transformed = np.array(preprocessor.transform(X_train))
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_train_transformed = remove_prefix(X_train_transformed)

    X_test_transformed = np.array(preprocessor.transform(X_test))
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
    X_test_transformed = remove_prefix(X_test_transformed)

    # Reset index
    X_train_transformed.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test_transformed.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    print("\t Making all columns numeric...")
    ## Make all columns numeric for ML models
    for col in X_train_transformed.columns:
        try:
            X_train_transformed[col] = pd.to_numeric(X_train_transformed[col])
        except Exception as e:
            print(f"Column {col} failed: {e}")

    for col in X_test_transformed.columns:
        try:
            X_test_transformed[col] = pd.to_numeric(X_test_transformed[col])
        except Exception as e:
            print(f"Column {col} failed: {e}")

    ### Save processed data ###
    if data_path:
        print("\t Saving processed data...")
        if nomo:
            data_path = data_path / "nomo"
        else:
            data_path = data_path / "base"
        if data_path.exists():
            warnings.warn(f"Over-writing tabular data at path: {data_path}")
            rmtree(data_path)
        data_path.mkdir(exist_ok=False, parents=True)

        ## Save transformed data
        X_train_transformed.to_parquet(data_path / "X_train.parquet")
        y_train.to_excel(data_path / "y_train.xlsx")
        X_test_transformed.to_parquet(data_path / "X_test.parquet")
        y_test.to_excel(data_path / "y_test.xlsx")
    ### Save fitted preprocessor/pipeline ###
    if pipeline_path:
        print("\t Saving pipeline...")
        if nomo:
            preprocessor_path = pipeline_path / "nomo_pipeline.joblib"
        else:
            preprocessor_path = pipeline_path / "base_pipeline.joblib"
        if preprocessor_path.exists():
            warnings.warn(f"Over-writing tabular data at path: {data_path}")
            preprocessor_path.unlink()
        preprocessor_path.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(preprocessor, preprocessor_path, compress=3)
    return {
        "X_train": X_train_transformed,
        "y_train": y_train,
        "X_test": X_test_transformed,
        "y_test": y_test,
    }
