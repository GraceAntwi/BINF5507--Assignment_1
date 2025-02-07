# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Impute Missing Values
def impute_missing_values(clean_data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    """
    if strategy == 'mean':
        numerical_col = clean_data.select_dtypes(include=["float64",'int64']).columns
        for col in numerical_col:
           clean_data[col].fillna(clean_data[col].mean(), inplace = True)
    elif strategy == 'median':
         numerical_col = clean_data.select_dtypes(include=["float64",'int64']).columns
         for col in numerical_col:
           clean_data[col].fillna(clean_data[col].median(), inplace = True)
    elif strategy == 'mode':
        for col in clean_data.columns:
            mode_series = clean_data[col].mode()
            if not mode_series.empty:
                clean_data[col].fillna(mode_series[0], inplace = True)
    else:
        raise ValueError('Strategy not identified')
    return clean_data
    pass

# 2. Remove Duplicates
def remove_duplicates(clean_data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    clean_data.drop_duplicates()
    return clean_data
    pass

# 3. Normalize Numerical Data
def normalize_data(clean_data, method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    """
    numerical_col = clean_data.select_dtypes(include=["float64",'int64']).columns
    if method == 'minmax':
        for col in numerical_col:
            min_val = clean_data[col].min()
            max_val = clean_data[col].max()
            if max_val - min_val != 0:
                clean_data[col] = (clean_data[col] - min_val) / (max_val - min_val)
            else:
                clean_data[col] = 0.0
    elif method == 'standard':
        for col in numerical_col:
            mean_val = clean_data[col].mean()
            std_val = clean_data[col].std()
            if std_val != 0:
                clean_data[0] = (clean_data[col] - mean_val) / std_val
            else:
                clean_data[col] = 0.0
    else:
        raise ValueError("Unsupported normalization methos")
    return clean_data
    pass

# 4. Remove Redundant Features   
def remove_redundant_features(clean_data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    """
    numeric_data = clean_data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr().abs()
    upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape), k=0) .astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return clean_data.drop(to_drop, axis=1)
    pass

# ---------------------------------------------------

def simple_model(clean_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    clean_data.dropna(inplace = True)

    # split the data into features and target
    target = clean_data.iloc[:, 0]
    features = clean_data.iloc[:, 1:]
    #converting target from float binary to integer binary to get log_reg to run
    target = target.astype(int)


    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        scaler = standardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None