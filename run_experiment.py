from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mlflow.models import infer_signature
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class XGBModelPipline:
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols

    def get_model(self, params):
        # build a pipeline
        ordinal_encoder = preprocessing.OrdinalEncoder()
        preprocess = ColumnTransformer(
            [("Ordinal-Encoder", ordinal_encoder, self.categorical_cols)],
            remainder="passthrough",
        )
        xgb_model = xgb.XGBRegressor(**params)
        model = Pipeline([("preprocess", preprocess), ("xgb_model", xgb_model)])
        return model

    def get_space(self):
        # the space for searching for the best parameter setting
        space = {
            "max_depth": hp.uniformint("max_depth", 2, 8),
            "subsample": hp.quniform("subsample", 0.5, 1, 0.1),
            "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-1)),
            "min_child_weight": hp.uniformint("min_child_weight", 1, 5),
        }
        return space


class HyperparameterOptimizer:
    def __init__(
        self,
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        experiment_name,
        artifact_path,
        registered_model_name,
        evaluator=mean_squared_error,
        evaluator_name="mse",
        max_evals=100,
    ):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.evaluator = evaluator
        self.evaluator_name = evaluator_name
        self.max_evals = max_evals
        self.experiment_name = experiment_name
        self.artifact_path = artifact_path
        self.registered_model_name = registered_model_name
        self.signature = infer_signature(train_x[:1], train_y[:1])

    def train_model(self, params):
        model = self.model.get_model(params)

        # Train model with MLflow tracking
        with mlflow.start_run(nested=True):
            _ = model.fit(self.train_x, self.train_y)

            # Evaluattion
            pred_y = model.predict(self.val_x)
            eval_metric = self.evaluator(self.val_y, pred_y)

            # Log the parameters, metric and model
            mlflow.log_params(params)
            mlflow.log_metric(self.evaluator_name, eval_metric)
            mlflow.sklearn.log_model(
                model, artifact_path=self.artifact_path, signature=self.signature
            )
            result = {"loss": eval_metric, "status": STATUS_OK, "model": model}
            return result

    def objective(self, params):
        # the objective function builds a new model for each parameter setting and MLFlow trackes parameters and results
        result = self.train_model(params)
        return result

    def run_experiment(self):
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            # Conduct the hyperparameter search using Hyperopt
            trials = Trials()
            best_parameters = fmin(
                fn=self.objective,
                space=self.model.get_space(),
                algo=tpe.suggest,
                max_evals=self.max_evals,
                trials=trials,
            )

            # find the best model
            best_trial = sorted(trials.results, key=lambda x: x["loss"])[0]

            # Log the best parameters, MSE, and model
            mlflow.log_params(best_parameters)
            mlflow.log_metric(self.evaluator_name, best_trial["loss"])

            best_model_info = mlflow.sklearn.log_model(
                best_trial["model"],
                artifact_path=self.artifact_path,
                signature=self.signature,
                input_example=self.train_x,
                registered_model_name=self.registered_model_name,
            )
        return best_parameters, best_trial, best_model_info.model_uri


def load_dataset(path):
    df = pd.read_csv(path)
    categorical_cols = ["make", "model", "fuel", "gear", "offerType"]
    numerical_cols = ["mileage_log", "hp", "age", "price_log"]

    cols = categorical_cols + numerical_cols
    data = df[cols]

    train, val_and_test = train_test_split(data, test_size=0.30, random_state=37)
    train_x = train.drop(["price_log"], axis=1)
    train_y = train[["price_log"]]

    val, test = train_test_split(val_and_test, test_size=0.50, random_state=37)
    val_x = val.drop(["price_log"], axis=1)
    val_y = val[["price_log"]]
    test_x = test.drop(["price_log"], axis=1)
    test_y = test[["price_log"]]
    return (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        categorical_cols,
        numerical_cols,
    )


if __name__ == "__main__":
    # setting the URI of the MLflow tracking server
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # reading the dataset
    german_car_dataset = None
    # german_car_dataset.load_dataset()

    dataset_path = Path("./data/autoscout24-germany-dataset-cleaned.csv")
    train_x, train_y, val_x, val_y, test_x, test_y, categorical_cols, _ = load_dataset(
        dataset_path
    )

    # defining the hyperparameter finder object and running the experiment
    hyperparmeter_finder = HyperparameterOptimizer(
        XGBModelPipline(categorical_cols),
        train_x,
        train_y,
        val_x,
        val_y,
        experiment_name="german-car-price",
        artifact_path="german_car_model",
        registered_model_name="german-car-price-best-model",
    )
    best_parameters, best_trail, model_uri = hyperparmeter_finder.run_experiment()

    print(f"Parameters of the best model: {best_parameters}")
    print(
        f"Mean squared error (MSE) of the best model on the validation dataset: {best_trail['loss']}"
    )

    # loading the best model and testing its performance with the test dataset
    loaded_model = mlflow.sklearn.load_model(model_uri)
    predictions = loaded_model.predict(test_x)
    test_mse = mean_squared_error(test_y, predictions)
    print(f"Mean squared error (MSE) of the best model on the test dataset: {test_mse}")
