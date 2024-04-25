from sklearn.model_selection import GridSearchCV, KFold,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# HYPERPARAMETERS

hyper_param_rf_regressor_pipeline = {
                                        "Estimator__n_estimators" : [4,5],
                                        "Estimator__criterion" : ["squared_error", "absolute_error"],
                                        "Estimator__max_depth" : [None],
                                        "Estimator__min_samples_split": [10,20],
                                        "Estimator__max_features" : ["sqrt","log2"],
                                      }

# METRICS

regression_metrics = ["neg_mean_absolute_error","neg_root_mean_squared_error","r2"]


# CV STRATEGIES
n_splits = 5
cv_strategy = [KFold(n_splits=n_splits),StratifiedKFold(n_splits=n_splits)]


# GRID RESEARCH CV
grid_search = GridSearchCV(
                                estimator = "objeto_pipeline",
                                param_grid = hyper_param_rf_regressor_pipeline,
                                scoring = regression_metrics ,
                                cv = KFold(n_splits= 5),
                                refit = "r2",
                                error_score = "raise"
                            )

"""                         
grid_search.fit(X_train,y_train)
grid_search.cv_results_
grid_search.best_params_
"""