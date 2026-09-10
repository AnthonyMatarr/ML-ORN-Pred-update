import os
import argparse
from pathlib import Path
import time
import optuna
import pandas as pd
import numpy as np
import logging
import json
import torch
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.config import SEED
from src.nn_model import TorchNNClassifier
from sklearn.metrics import roc_auc_score
import warnings

N_SPLITS = 5


################################################################
##################### Train/Prelim Results #####################
################################################################
def train_and_prelim_eval(data_dict, json_path, model_save_path=None):
    """
    Train neural network with best hyperparameters and evaluate on validation set.

    Parameters
    ----------
    X_train_path : Path
        Path to training features
    y_train_path : Path
        Path to training labels
    params_json_path : Path
        Path to JSON file with best hyperparameters
    """
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"].values.ravel()
    with open(json_path, "r") as f:
        json_params = json.load(f)
    print(f"Training nn for...")
    print(f"\t Best CV AUROC: \t{json_params["best_score"]:.3f}")
    # ==================================> Extract best param
    best_params = json_params["best_params"]
    # Extract architecture parameters
    hidden_sizes = [
        best_params["hl_1"],
    ]
    dropouts = [best_params["dr_1"]]
    if "hl_2" in best_params:
        hidden_sizes.append(best_params["hl_2"])
        dropouts.append(best_params["dr_2"])
    # ============================> Create/fit model
    clf = TorchNNClassifier(
        hidden_size_list=hidden_sizes,
        dropouts=dropouts,
        activation_name=best_params["activation"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        optimizer_str=best_params["optimizer"],
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        device="cpu",
        seed=SEED,
    )
    clf.fit(X_train, y_train)
    # ===================================> Evaluate
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_proba)
    ##### Output results #####
    print(f"Train AUROC: \t{train_auc:.3f}")
    if "X_test" in data_dict and "y_test" in data_dict:
        X_test = data_dict["X_test"]
        y_test = data_dict["y_test"].values.ravel()
        y_test_proba = clf.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_proba)
        print(f"Test AUROC: \t{test_auc:.3f}")
    print(f"PARAMS: \t{json_params["best_params"]}")
    print("*" * 10)
    # ==================================> Export
    # Prepare hyperparameters dictionary for saving
    h_params_to_save = {
        "hl_1": best_params["hl_1"],
        "dr_1": best_params["dr_1"],
        "act_func_str": best_params["activation"],
        "num_epochs": best_params["epochs"],
        "lr": best_params["lr"],
        "weight_decay": best_params["weight_decay"],
        "batch_size": best_params["batch_size"],
        "optimizer_str": best_params["optimizer"],
        "weight_init_scheme": best_params["weight_init"],
    }
    # Add third layer if present
    if "hl_2" in best_params:
        h_params_to_save["hl_2"] = best_params["hl_2"]
        h_params_to_save["dr_2"] = best_params["dr_2"]
    # Create checkpoint dictionary
    checkpoint = {
        "h_params": h_params_to_save,
        "state_dict": clf.model_.state_dict(),  # type: ignore
        "feature_names_in_": clf.feature_names_in_,
    }
    # Save to file
    if model_save_path:
        if model_save_path.exists():
            warnings.warn(f"Over-writing saved model at path {model_save_path}")
            model_save_path.unlink()
        model_save_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(checkpoint, model_save_path)
        print(f"Model saved to {model_save_path}")


################################################################
######################## Model Builders ########################
################################################################
# ======================================================> STAGE 1
def nn_model_builder(trial):
    # Network architecture (1-2 layers)
    n_layers = trial.suggest_int("n_layers", 1, 2)

    hl_1 = trial.suggest_int("hl_1", 16, 256)
    dr_1 = trial.suggest_float("dr_1", 0.0, 0.5)

    hidden_size_list = [hl_1]
    dropouts = [dr_1]

    if n_layers == 2:
        hl_2 = trial.suggest_int("hl_2", 8, 64)
        dr_2 = trial.suggest_float("dr_2", 0.0, 0.5)
        hidden_size_list.append(hl_2)
        dropouts.append(dr_2)

    # Activation function
    activation_name = trial.suggest_categorical(
        "activation", ["relu", "leaky_relu", "elu"]
    )

    # Training hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int(
        "epochs", 50, 300
    )  # Tune epochs instead of early stopping

    optimizer_str = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    weight_init_scheme = trial.suggest_categorical(
        "weight_init", ["xavier_uniform", "kaiming_uniform"]
    )

    return TorchNNClassifier(
        hidden_size_list=hidden_size_list,
        dropouts=dropouts,
        activation_name=activation_name,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_str=optimizer_str,
        epochs=epochs,
        batch_size=batch_size,
        weight_init_scheme=weight_init_scheme,
        verbose=0,
        seed=SEED,
    )


################################################################
######################### More Set Up ##########################
################################################################
def build_parser():
    parser = argparse.ArgumentParser(
        prog="Neural Network Tuner",
        description="Tune a neural network with optuna",
    )
    parser.add_argument(
        "--X_path", required=True, help="String path to embedding file (X data)"
    )
    parser.add_argument(
        "--y_path",
        required=True,
        help="String path to cleaned raw tabbular data (used to generate y data)",
    )
    parser.add_argument(
        "--gpu_ids_str",
        required=True,
        help='Comma-separated list of GPU IDs (e.g., `"0,1"`, `"4,5,6"`)',
    )
    parser.add_argument(
        "--scoring_str",
        required=True,
        help='Scoring metric used during cross-validation (e.g., `"roc_auc"`)',
    )
    parser.add_argument(
        "--log_path",
        required=True,
        help="Log file path where tuning information for this outcome will be written.",
    )
    parser.add_argument(
        "--results_path",
        required=True,
        help="Output file path where the best score and best trial parameters will be saved in JSON format.",
    )
    parser.add_argument(
        "--n_trials",
        required=True,
        type=int,
        help=f"Number of Optuna trials to run during optimization.",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help=f"Random seed used for Optuna samplers and StratifiedKFold splitting to ensure reproducibility.",
    )
    return parser


def load_data(X_path, y_path):
    """
    Load/subset data from memory

    perc: float
        % of training data to use--> anywhere from (0.0, 1.0]
    """
    X_train = pd.read_parquet(X_path)
    y_train_imp = pd.read_excel(y_path, index_col=0)
    y_train = y_train_imp.values.ravel()
    return X_train, y_train


def parse_arguments(argv=None):
    return build_parser().parse_args(argv)


def main_tuner(
    *_,
    X_path,
    y_path,
    gpu_ids_str,
    scoring_str,
    log_path,
    results_path,
    n_trials,
    rand_state,
):
    """
    Tunes a neural network classifier for a single binary outcome using Optuna.
    This function is designed to be called as a standalone tuning worker, typically
    launched in parallel with different GPU assignments. The function loads data,
    configures logging, restricts visible GPUs, constructs a device-aware model
    builder, runs cross-validated optimization, and writes tuning logs and results
    to disk.

    Parameters
    ----------
    X_path : pathlib.Path
        File path to the predictor matrix used for tuning

    y_path : pathlib.Path
        File path to the outcome data

    gpu_ids_str : str
        Comma-separated list of GPU IDs to expose to this process via the
        `CUDA_VISIBLE_DEVICES` environment variable (e.g., `"0,1"`, `"4,5,6"`).
        Determines which GPUs are available to the model during training.

    scoring_str : str
        Scoring metric used during cross-validation (e.g., `"roc_auc"`). Passed to
        `cross_val_score()`.

    log_path : pathlib.Path
        Log file path where tuning information for this outcome will be written.

    results_path : pathlib.Path
        Output file path where the best score and best trial parameters will be
        saved in JSON format.

    n_trials : int
        Number of Optuna trials to run during optimization.

    rand_state : int
        Random seed used for Optuna samplers and StratifiedKFold splitting to
        ensure reproducibility.

    Returns
    -------
    None
        This function performs tuning, logging, and writing of results to disk.
        It does not return any objects.

    Raises
    ------
    ValueError
        If positional arguments are supplied (this function only accepts keyword
        arguments).

    FileNotFoundError
        If the provided `X_path` or `y_path` does not exist or cannot be loaded by
        `load_data()`.

    RuntimeError
        If CUDA is expected but not available, or if model training fails due to
        GPU configuration or resource errors.

    Notes
    -----
    - The model architecture is constructed inside the objective function using a
      wrapped builder that enforces a specific GPU device.
    - GPU visibility is controlled exclusively by `CUDA_VISIBLE_DEVICES`; within
      this restricted namespace, the classifier uses `"cuda"` automatically.
    - Logging is written incrementally throughout tuning to the designated log file.
    - Each outcome produces one result JSON file containing the best score and
      hyperparameter set.
    """
    if _ != tuple():
        raise ValueError("main_tuner() does not accept positional arguments!")
    # ============ Configure Logger ================
    if log_path.exists():
        log_path.unlink()
    log_path.parent.mkdir(exist_ok=True, parents=True)
    file_handler = logging.FileHandler(log_path, mode="a")
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    # ============ Configure Device ================
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logging.info(f"Starting tuning...")
    logging.info(f"CUDA_VISIBLE_DEVICES={gpu_ids_str}, device={device}")

    def build_nn_estimator_for_device(trial):
        """
        Model builder wrapper to set device
        Model created at train time, so setting device after initialization is sufficient
        """
        nn_clf = nn_model_builder(trial)
        nn_clf.device = device
        nn_clf.seed = SEED
        return nn_clf

    logging.info(f"Loading data...")
    X_train, y_train = load_data(X_path, y_path)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=rand_state)

    def objective(trial):
        """
        Creates objective for optuna tuning
        """
        start_time = time.perf_counter()
        logging.info(f"Starting trial {trial.number}")
        clf = build_nn_estimator_for_device(trial)
        scores = cross_val_score(
            clf,
            X_train,
            y_train,
            scoring=scoring_str,
            cv=skf,
            n_jobs=1,  # let GPU handle parallelism
        )
        mean_score = float(np.mean(scores))
        # ----- timing + logging end -----
        elapsed = time.perf_counter() - start_time
        logging.info(
            f"Finished trial {trial.number}: "
            f"score={mean_score:.4f}, "
            f"time={elapsed/60:.2f} min"
        )
        logging.info(f"Trial {trial.number}: score={mean_score:.4f}")
        return mean_score

    logging.info(f"Creating study...")
    # ============ Tune ================
    study_name = f"nn_study"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=rand_state),
        pruner=optuna.pruners.HyperbandPruner(min_resource=5, reduction_factor=3),
    )
    logging.info(f"Optuna study created: {study_name}")
    study.optimize(objective, n_trials=n_trials)

    # ====== Deal with results ======
    ## Log results
    logging.info(f"Best {scoring_str}: {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_params}")
    ## Save results
    result = {
        "best_score": round(study.best_value, 4),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    ## Each outcome will get its own file
    if results_path.exists():
        results_path.unlink()
    results_path.parent.mkdir(exist_ok=True, parents=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=4)

    logging.info(f"Saved best results to {results_path}")

    # ====== Clean up ======
    root_logger.removeHandler(file_handler)
    file_handler.close()


########################################################
######################### Main #########################
########################################################
def main(argv=None):
    ## Parse args
    args = parse_arguments(argv)
    ## Convert str paths to pathlib Paths
    X_path = Path(args.X_path)
    y_path = Path(args.y_path)
    log_path = Path(args.log_path)
    results_path = Path(args.results_path)
    main_tuner(
        X_path=X_path,
        y_path=y_path,
        gpu_ids_str=args.gpu_ids_str,
        scoring_str=args.scoring_str,
        log_path=log_path,
        results_path=results_path,
        n_trials=args.n_trials,
        rand_state=args.seed,
    )


if __name__ == "__main__":
    main()
