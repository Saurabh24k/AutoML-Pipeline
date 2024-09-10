from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

# Smart model selection for classification and regression
def smart_model_selection(df, target_col, task_type='classification', hyperparameter_tuning=False):
    """
    Smartly selects a model based on task type (classification or regression) and performs hyperparameter tuning if required.
    
    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        target_col (str): The name of the target column.
        task_type (str): Task type - 'classification' or 'regression'.
        hyperparameter_tuning (bool): Whether to apply hyperparameter tuning (True or False).
        
    Returns:
        best_model: The best model after selection and tuning.
        best_params: The best hyperparameters, if tuning is applied.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Check if target variable is appropriate
    if y.nunique() <= 1:
        raise ValueError(f"The target column '{target_col}' needs to have more than one unique value.")

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        raise ValueError(f"Error in train-test split: {e}")

    best_model = None
    best_params = None

    # Classification Task
    if task_type == 'classification':
        # Use RandomForestClassifier and SVC as baseline models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(),
            "Support Vector Classifier": SVC(),
        }
        
        best_model, best_score = None, 0
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                if score > best_score:
                    best_model, best_score = model, score
            except Exception as e:
                print(f"Error training {name}: {e}")

        if best_model is None:
            raise ValueError("No model could be successfully trained.")

        # Hyperparameter tuning
        if hyperparameter_tuning:
            param_grid = {
                'Random Forest Classifier': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None]
                },
                'Support Vector Classifier': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            }
            
            best_model_name = type(best_model).__name__
            if best_model_name in param_grid:
                try:
                    search = GridSearchCV(best_model, param_grid[best_model_name], cv=3)
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                except Exception as e:
                    print(f"Error during hyperparameter tuning for {best_model_name}: {e}")

        # Classification evaluation
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        return best_model, {"accuracy": accuracy, "f1_score": f1}, best_params

    # Regression Task
    elif task_type == 'regression':
        # Use LinearRegression, RandomForestRegressor, and SVR as baseline models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Support Vector Regressor": SVR(),
        }
        
        best_model, best_score = None, float('inf')
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = mean_squared_error(y_test, y_pred)
                if score < best_score:
                    best_model, best_score = model, score
            except Exception as e:
                print(f"Error training {name}: {e}")

        if best_model is None:
            raise ValueError("No regression model could be successfully trained.")

        # Hyperparameter tuning
        if hyperparameter_tuning:
            param_grid = {
                'Random Forest Regressor': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None]
                },
                'Support Vector Regressor': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            }
            
            best_model_name = type(best_model).__name__
            if best_model_name in param_grid:
                try:
                    search = GridSearchCV(best_model, param_grid[best_model_name], cv=3)
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                except Exception as e:
                    print(f"Error during hyperparameter tuning for {best_model_name}: {e}")

        # Regression evaluation
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return best_model, {"mae": mae, "mse": mse, "r2_score": r2}, best_params

    else:
        raise ValueError(f"Unknown task type '{task_type}'. Supported types are 'classification' and 'regression'.")



# Model Evaluation Function
def evaluate_model(model, X_test, y_test, task_type='classification'):
    """
    Evaluates the model based on the task type and returns the evaluation metrics.
    
    Parameters:
        model: Trained model to evaluate.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target variable.
        task_type (str): Task type - 'classification' or 'regression'.
        
    Returns:
        metrics (dict): Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)

    if task_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return {"accuracy": accuracy, "f1_score": f1}
    
    elif task_type == 'regression':
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"mae": mae, "mse": mse, "r2_score": r2}
