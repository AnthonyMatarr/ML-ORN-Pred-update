from src.config import SEED

## General purpose
import joblib
import json
import warnings
import logging
import numpy as np
import optuna
from sklearn.exceptions import ConvergenceWarning

## Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.base import ClassifierMixin, BaseEstimator

## Sampling
from imblearn.over_sampling import ADASYN, SMOTENC
from imblearn.pipeline import Pipeline
from src.data_utils import get_feature_lists

N_SPLITS = 5
N_PARALLEL_CV_JOBS = N_SPLITS


#############################################################################################
###################################### MODEL BUILDERS #######################################
#############################################################################################
# ========================> LOGISTIC REGRESSION
def lr_model_builder(trial):
    C = trial.suggest_float("C", 1e-3, 5.0, log=True)
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])

    if penalty == "elasticnet":
        solver = "saga"
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    elif penalty == "l1":
        solver = trial.suggest_categorical("solver_l1", ["liblinear", "saga"])
        l1_ratio = None
    else:  # l2
        solver = trial.suggest_categorical(
            "solver_l2", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
        )
        l1_ratio = None

    # Moderately constrained pos_weight
    if solver in ["saga", "sag"]:
        pos_weight = trial.suggest_float("pos_weight_saga", 1.0, 8.0, log=True)
    elif solver == "liblinear":
        pos_weight = trial.suggest_float("pos_weight_linear", 1.0, 12.0, log=True)
    else:
        pos_weight = trial.suggest_float("pos_weight_general", 1.0, 8.0, log=True)

    class_weight = {0: 1.0, 1: pos_weight}

    if solver == "liblinear":
        intercept_scaling = trial.suggest_float(
            "intercept_scaling", 1e-2, 1e2, log=True
        )
    else:
        intercept_scaling = 1.0

    max_iter = 5000 if solver in ["saga", "sag"] else 4000

    return LogisticRegression(
        penalty=penalty,
        C=C,
        tol=1e-4,
        fit_intercept=True,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=SEED,
        solver=solver,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        warm_start=False,
        n_jobs=1,
    )


# ========================> LIGHTGBM
def lightgbm_model_builder(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
    n_estimators = trial.suggest_int("n_estimators", 100, 800)

    max_depth = trial.suggest_int("max_depth", 3, 6)
    max_leaves = min(64, 2**max_depth)
    min_leaves = min(8, max_leaves)
    num_leaves = trial.suggest_int("num_leaves", min_leaves, max_leaves)

    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 30, 100)
    min_gain_to_split = trial.suggest_float("min_gain_to_split", 0.5, 5.0)

    feature_fraction = trial.suggest_float("feature_fraction", 0.5, 0.9)
    bagging_fraction = trial.suggest_float("bagging_fraction", 0.6, 0.9)
    bagging_freq = trial.suggest_int("bagging_freq", 1, 5)
    scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 5.0, log=True)

    lambda_l1 = trial.suggest_float("lambda_l1", 1.0, 25.0)
    lambda_l2 = trial.suggest_float("lambda_l2", 1.0, 25.0)

    max_bin = trial.suggest_int("max_bin", 64, 264)

    return LGBMClassifier(
        objective="binary",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        min_split_gain=min_gain_to_split,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        scale_pos_weight=scale_pos_weight,
        max_bin=max_bin,
        tree_learner="feature_parallel",
        n_jobs=1,
        seed=SEED,
        bagging_seed=SEED,
        feature_fraction_seed=SEED,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )


# ========================> XGBOOST
def xgb_model_builder(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500)

    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    max_depth = trial.suggest_int("max_depth", 2, 10)

    gamma = trial.suggest_float("gamma", 1.0, 10.0)
    reg_alpha = trial.suggest_float("reg_alpha", 5.0, 50.0)
    reg_lambda = trial.suggest_float("reg_lambda", 5.0, 50.0)

    subsample = trial.suggest_float("subsample", 0.5, 0.8)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 0.8)
    colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 0.9)

    scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 15.0, log=True)

    min_child_weight = trial.suggest_int("min_child_weight", 5, 30)

    return XGBClassifier(
        objective="binary:logistic",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=min_child_weight,
        tree_method="hist",
        n_jobs=1,
        random_state=SEED,
        eval_metric="auc",
    )


# ========================> SVC
def svc_model_builder(trial):
    C = trial.suggest_float("C", 1e-3, 10, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])

    if kernel == "poly":
        degree = trial.suggest_int("degree", 2, 5)
    else:
        degree = 3  # default

    if kernel in ["rbf", "poly", "sigmoid"]:
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    else:
        gamma = "scale"
    pos_weight = trial.suggest_float("pos_weight", 1, 10)
    max_iter = trial.suggest_int("max_iter", 1000, 10000) if kernel == "rbf" else 2000
    model = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        class_weight={0: 1, 1: pos_weight},
        probability=True,
        random_state=SEED,
        max_iter=max_iter,
    )
    return model


# ========================> KNN
def knn_model_builder(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 3, 12)
    weights = "distance"
    algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree"])
    leaf_size = trial.suggest_int("leaf_size", 10, 35)
    p = trial.suggest_int("p", 1, 2)  # 1=manhattan, 2=euclidean

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        n_jobs=1,
    )
    return model


#############################################################################################
########################################## TUNING ###########################################
#############################################################################################
def get_sampler(strategy, ratio, n_neighbors, cat_feats):
    """
    Helper function that returns sampler
    Parameters
    ----------
    strategy: string; One of ['smotenc', 'adasyn')
        Specify sampling strategy to use
    ratio: float
        Sampling ratio used within appropriate sampler
    """
    if strategy == "adasyn":
        sampler = ADASYN(
            sampling_strategy=ratio,
            random_state=SEED,
            n_neighbors=n_neighbors,
        )
    elif strategy == "smotenc":
        sampler = SMOTENC(
            categorical_features=cat_feats,
            sampling_strategy=ratio,
            random_state=SEED,
            k_neighbors=n_neighbors,
        )
    else:
        raise ValueError(f"Got unknown sampling strategy: {strategy}")
    return sampler


def make_objective(X_train, y_train, model_builder, scoring="roc_auc"):
    """
    Creates objective for model tuning
    Includes sampling param and calls appropriate model builder
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    def objective(trial):
        sampling_strategy = trial.suggest_categorical(
            "sampling_strategy", ["adasyn", "none", "smotenc"]
        )
        if sampling_strategy != "none":
            sampling_ratio = trial.suggest_float("sampling_ratio", 0.5, 1.0)
            n_neighbors = trial.suggest_int("sampling_neighbors", 3, 8)
        else:
            n_neighbors = None
            sampling_ratio = None
        # build model
        model = model_builder(trial)
        # create pipeline
        if sampling_strategy == "none":
            pipeline = model
        else:
            num_cols = get_feature_lists(X_train)["Numerical"]
            cat_feats = [col for col in X_train.columns if col not in num_cols]
            ## Create pipeline w/ sampler
            sampler = get_sampler(
                strategy=sampling_strategy,
                ratio=sampling_ratio,
                n_neighbors=n_neighbors,
                cat_feats=cat_feats,
            )
            pipeline = Pipeline([("sampler", sampler), ("classifier", model)])
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring=scoring,
            cv=skf,
            n_jobs=N_PARALLEL_CV_JOBS,
        )
        return np.round(np.mean(scores), 4)

    return objective


def tune_single_model(
    model_builder,
    model_name,
    X_train,
    y_train,
    scoring="roc_auc",
    log_file_path=None,
    save_path=None,
    n_trials=100,
    clear_progress=False,
):
    """
    Tunes a single model using Optuna and writes params/tuning results to disk.

    Parameters
    ----------
    model_builder: callable
        Function that takes an optuna.trial object and returns a built estimator
    model_name: str
        Name of model to be tuned (e.g., 'LogisticRegression', 'XGBoost')
    X_train: pd.DataFrame or np.ndarray
        Training features
    y_train: pd.Series or np.ndarray
        Training labels
    scoring: str
        Scoring metric for tuning (passed to cross_val_score)
    log_file_path: pathlib.Path, optional
        Path where tuning logs will be written
    save_path: pathlib.Path, optional
        Path where best CV score/params are written in JSON format
    n_trials: int
        Number of Optuna trials to run
    clear_progress: bool
        If True, delete existing study database before starting

    Returns
    -------
    dict
        Dictionary containing best_score, best_params, and study object
    """
    file_handler = None
    root_logger = None
    ############ Set up paths ############
    if log_file_path:
        if log_file_path.exists():
            warnings.warn(f"Over-writing log at path: {log_file_path}")
            log_file_path.unlink()
        log_file_path.parent.mkdir(exist_ok=True, parents=True)

    if save_path:
        save_path.parent.mkdir(exist_ok=True, parents=True)

        # Clear existing study database
        db_file = save_path.parent / f"{model_name}.db"
        if clear_progress and db_file.exists():
            db_file.unlink()
            if log_file_path:
                with open(log_file_path, "a") as f:
                    f.write(f"Deleted existing study database: {db_file.name}\n")

    ############ Set up logger ############
    if log_file_path:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Remove existing FileHandlers
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
                handler.close()

        # Add new handler
        file_handler = logging.FileHandler(log_file_path, mode="a")
        root_logger.addHandler(file_handler)

        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()

    ############ Prepare data ############
    if hasattr(y_train, "values"):
        y_train = y_train.values.ravel()
    else:
        y_train = np.asarray(y_train).ravel()

    ############ Run Optuna study ############
    if log_file_path:
        with open(log_file_path, "a") as f:
            f.write(f"Starting tuning for: {model_name}\n")
            f.write(f"Training samples: {len(y_train)}, Features: {X_train.shape[1]}\n")

    # Create objective
    objective = make_objective(X_train, y_train, model_builder, scoring)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        if save_path:
            storage = f"sqlite:///{save_path.parent / f'{model_name}.db'}"
        else:
            storage = None

        study = optuna.create_study(
            storage=storage,
            study_name=f"{model_name}_study",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.HyperbandPruner(),
            load_if_exists=True,
        )

        # Run optimization
        study.optimize(objective, n_trials=n_trials, n_jobs=1)  # type: ignore

    ############ Extract results ############
    try:
        best_score = study.best_value
        best_params = study.best_params

        # Separate sampling params from model params
        sampling_strategy = best_params.get("sampling_strategy", "none")
        sampling_ratio = best_params.get("sampling_ratio", None)
        sampling_neighbors = best_params.get("sampling_neighbors", None)

        # Extract model-specific params (exclude sampling params)
        model_params = {
            k: v
            for k, v in best_params.items()
            if k not in ["sampling_strategy", "sampling_ratio", "sampling_neighbors"]
        }

        result_dict = {
            "model_name": model_name,
            "best_score": float(best_score),
            "best_params": best_params,  # All params combined
            "model_params": model_params,  # Model params only
            "sampling_params": {
                "strategy": sampling_strategy,
                "ratio": sampling_ratio,
                "n_neighbors": sampling_neighbors,
            },
            "n_trials": len(study.trials),
            "n_train": len(y_train),
            "n_features": X_train.shape[1],
            "scoring_metric": scoring,
            "n_cv_folds": N_SPLITS,
        }

        if log_file_path:
            with open(log_file_path, "a") as f:
                f.write(f"\n{'='*100}\n")
                f.write(f"TUNING COMPLETE FOR {model_name}\n")
                f.write(f"{'='*100}\n")
                f.write(f"Best {scoring}: {best_score:.4f}\n\n")
                f.write(f"Model parameters:\n")
                for k, v in model_params.items():
                    f.write(f"  {k}: {v}\n")
                f.write(f"\nSampling configuration:\n")
                f.write(f"  Strategy: {sampling_strategy}\n")
                if sampling_strategy != "none":
                    f.write(f"  Ratio: {sampling_ratio:.3f}\n")
                    f.write(f"  Neighbors: {sampling_neighbors}\n")
                f.write(f"\nCompleted {len(study.trials)} trials\n")
                f.write(f"{'='*100}\n")

    except ValueError as e:
        # All trials were pruned
        if log_file_path:
            with open(log_file_path, "a") as f:
                f.write(f"ERROR: All trials were pruned - {str(e)}\n")

        result_dict = {
            "model_name": model_name,
            "best_score": None,
            "best_params": {},
            "model_params": {},
            "sampling_params": {
                "strategy": None,
                "ratio": None,
                "n_neighbors": None,
            },
            "n_trials": len(study.trials),
            "n_train": len(y_train),
            "n_features": X_train.shape[1],
            "scoring_metric": scoring,
            "n_cv_folds": N_SPLITS,
            "error": "all_trials_pruned",
        }

    ############ Save results ############
    if save_path:
        with open(save_path, "w") as f:
            json.dump(result_dict, f, indent=4)

    # Clean up logger
    if log_file_path and root_logger and file_handler:
        root_logger.removeHandler(file_handler)
        file_handler.close()

    return result_dict


#############################################################################################
###################################### PRELIM RESULTS #######################################
#############################################################################################
def train_final_model(
    results_path,
    model_builder,
    model_name,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    model_save_path=None,
):
    """
    Trains final model using best hyperparameters from tuning results.

    Parameters
    ----------
    results_path: pathlib.Path
        Path to JSON file containing tuning results
    model_builder: callable
        Function that takes an optuna.trial object and returns a built estimator
    model_name: str
        Name of the model (e.g., 'knn', 'lr', 'xgb')
    X_train: pd.DataFrame or np.ndarray
        Training features
    y_train: pd.Series or np.ndarray
        Training labels
    X_test: pd.DataFrame or np.ndarray, optional
        Test features for evaluation
    y_test: pd.Series or np.ndarray, optional
        Test labels for evaluation
    model_save_path: pathlib.Path, optional
        Path to save the trained model. If None, model won't be saved.

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': trained model (or pipeline if sampling used)
        - 'train_auc': training AUROC
        - 'test_auc': test AUROC (if test data provided)
        - 'best_params': best hyperparameters from tuning
    """

    print(f"{'-'*30} Training {model_name.upper()} {'-'*30}")

    # Load tuning results
    with open(results_path, "r") as f:
        results = json.load(f)

    # Extract info
    best_score = results["best_score"]
    best_params = results["best_params"]
    model_params = results["model_params"]
    sampling_params = results["sampling_params"]

    sampling_strategy = sampling_params["strategy"]
    sampling_ratio = sampling_params["ratio"]
    sampling_neighbors = sampling_params["n_neighbors"]

    # Prepare y_train
    if hasattr(y_train, "values"):
        y_train_array = y_train.values.ravel()
    else:
        y_train_array = np.asarray(y_train).ravel()

    print(f"Best CV AUROC: {best_score:.4f}")
    print(f"Sampling: {sampling_strategy}", end="")
    if sampling_strategy != "none":
        print(f" (ratio={sampling_ratio:.3f}, neighbors={sampling_neighbors})")
    else:
        print(" (no resampling)")

    # Build model using FixedTrial
    trial = optuna.trial.FixedTrial(model_params)
    model = model_builder(trial)
    num_cols = get_feature_lists(X_train)["Numerical"]
    cat_cols = [col for col in X_train.columns if col not in num_cols]
    # Apply sampling if needed
    if sampling_strategy != "none":
        sampler = get_sampler(
            strategy=sampling_strategy,
            ratio=sampling_ratio,
            n_neighbors=sampling_neighbors,
            cat_feats=cat_cols,
        )
        X_train_resampled, y_train_resampled = sampler.fit_resample(  # type: ignore
            X_train, y_train_array
        )

        print(f"Train size: {len(y_train_resampled)} (original: {len(y_train_array)})")

        # Create pipeline for consistent predict interface
        from imblearn.pipeline import Pipeline

        pipeline = Pipeline([("sampler", sampler), ("classifier", model)])
        # Fit on original data (pipeline handles resampling)
        pipeline.fit(X_train, y_train_array)
        final_model = pipeline

    else:
        print(f"Train size: {len(y_train_array)}")
        model.fit(X_train, y_train_array)
        final_model = model
        X_train_resampled = X_train
        y_train_resampled = y_train_array

    # Calculate training AUC
    train_pred_proba = final_model.predict_proba(X_train)[:, 1]

    train_auc = roc_auc_score(y_train_array, train_pred_proba)
    print(f"Train AUROC: {train_auc:.4f}")

    # Calculate test AUC if provided
    test_auc = None
    if X_test is not None and y_test is not None:
        if hasattr(y_test, "values"):
            y_test_array = y_test.values.ravel()
        else:
            y_test_array = np.asarray(y_test).ravel()

        if model_name == "svc":
            test_pred_proba = final_model.decision_function(X_test)
        else:
            test_pred_proba = final_model.predict_proba(X_test)[:, 1]

        test_auc = roc_auc_score(y_test_array, test_pred_proba)
        print(f"Test AUROC:  {test_auc:.4f}")

    # Save model if path provided
    if model_save_path:
        model_save_path.parent.mkdir(exist_ok=True, parents=True)
        if model_save_path.exists():
            model_save_path.unlink()
        joblib.dump(final_model, model_save_path)
        print(f"Model saved to: {model_save_path}")

    print(f"Best params: {model_params}")
    print(f"{'='*80}\n")

    return {
        "model": final_model,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "best_params": best_params,
        "best_cv_score": best_score,
    }
