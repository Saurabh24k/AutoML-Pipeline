import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, KBinsDiscretizer, PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Custom Missing Value Handling
def handle_missing_values(df: pd.DataFrame, columns: list, strategy: str = 'mean'):
    """
    Handles missing values based on the selected columns and the user-chosen strategy.
    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        columns (list): The columns to process.
        strategy (str): The strategy to apply ('mean', 'median', 'mode').
    Returns:
        pd.DataFrame: The processed dataset.
    """
    for col in columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == 'object':
                if strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0])
    return df

# Custom Normalization/Standardization
def normalize_data(df: pd.DataFrame, columns: list, method: str = 'standard'):
    """
    Normalizes or standardizes the selected columns.
    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        columns (list): The columns to normalize/standardize.
        method (str): The method to apply ('standard' or 'minmax').
    Returns:
        pd.DataFrame: The processed dataset.
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Custom Categorical Encoding
def encode_categorical(df: pd.DataFrame, columns: list, method: str = 'onehot'):
    """
    Encodes categorical columns based on the selected method.
    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        columns (list): The columns to encode.
        method (str): The encoding method ('onehot' or 'label').
    Returns:
        pd.DataFrame: The processed dataset.
    """
    for col in columns:
        if method == 'onehot':
            df = pd.get_dummies(df, columns=[col], drop_first=True)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    return df

# Custom Outlier Handling
def handle_outliers(df: pd.DataFrame, columns: list, strategy: str = 'iqr'):
    """
    Detects and handles outliers based on the selected strategy (IQR).
    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        columns (list): The columns to process.
        strategy (str): The strategy to use ('iqr').
    Returns:
        pd.DataFrame: The processed dataset.
    """
    for col in columns:
        if strategy == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

# Custom Feature Selection
def select_features(df: pd.DataFrame, target_col: str, n_features: int = 5):
    """
    Dynamically selects the top N features using RFE.
    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        target_col (str): The name of the target column.
        n_features (int): Number of features to select.
    Returns:
        pd.DataFrame: The dataset with selected features.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model = RandomForestClassifier(n_estimators=100)
    rfe = RFE(model, n_features_to_select=n_features)
    fit = rfe.fit(X, y)
    selected_columns = X.columns[fit.support_]
    return df[selected_columns].join(df[target_col])

# Custom Binning/Discretization
def bin_columns(df: pd.DataFrame, columns: list, n_bins: int = 5):
    """
    Bins numerical columns into discrete intervals.
    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        columns (list): The columns to bin.
        n_bins (int): The number of bins to apply.
    Returns:
        pd.DataFrame: The processed dataset.
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    df[columns] = discretizer.fit_transform(df[columns])
    return df

# Custom SMOTE for Imbalanced Data
def handle_imbalance(df: pd.DataFrame, target_col: str):
    """
    Applies SMOTE to balance the classes.
    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        target_col (str): The name of the target column.
    Returns:
        pd.DataFrame: The processed dataset with balanced classes.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    resampled_df = pd.DataFrame(X_res, columns=X.columns)
    resampled_df[target_col] = y_res
    return resampled_df

# Polynomial Feature Generation
def generate_polynomial_features(df: pd.DataFrame, columns: list, degree: int = 2):
    """
    Adds polynomial features for the selected columns.
    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        columns (list): The columns to apply polynomial transformation.
        degree (int): Degree of polynomial features.
    Returns:
        pd.DataFrame: The dataset with added polynomial features.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[columns])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(columns))
    return poly_df.join(df.drop(columns=columns))

import pandas as pd

# Function to suggest columns to drop
def suggest_columns_to_drop(df: pd.DataFrame, missing_threshold: float = 0.5):
    """
    Suggests columns to drop based on criteria such as high cardinality, constant values, and high missing percentage.
    
    Parameters:
        df (pd.DataFrame): The dataset to analyze.
        missing_threshold (float): The threshold for missing values percentage to suggest dropping a column.
    
    Returns:
        List of suggested columns to drop.
    """
    suggested_columns = []

    # High cardinality columns (e.g., ID columns)
    high_cardinality_cols = [col for col in df.columns if df[col].nunique() > df.shape[0] * 0.8]
    suggested_columns.extend(high_cardinality_cols)

    # Constant or low variance columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    suggested_columns.extend(constant_cols)

    # Columns with high percentage of missing values
    missing_value_cols = [col for col in df.columns if df[col].isnull().mean() > missing_threshold]
    suggested_columns.extend(missing_value_cols)

    return list(set(suggested_columns))

# Function to drop selected columns
def drop_columns(df: pd.DataFrame, columns_to_drop: list):
    """
    Drops the selected columns from the dataset.
    
    Parameters:
        df (pd.DataFrame): The dataset to process.
        columns_to_drop (list): The list of columns to drop.
    
    Returns:
        pd.DataFrame: The dataset with the specified columns dropped.
    """
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df
