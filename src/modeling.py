# modeling.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
   classification_report,
   confusion_matrix,
   roc_auc_score,
   roc_curve,
   accuracy_score,
   f1_score,
   precision_score,
   recall_score,
)


def train_and_track_models(
   data_path,
   target_col,
   id_col,
   model_dir,
   report_dir,
   mlflow_experiment_name="default_experiment",
   random_state=42,
):
   os.makedirs(model_dir, exist_ok=True)
   os.makedirs(report_dir, exist_ok=True)
   df = pd.read_csv(data_path)
   print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")
   X = df.drop(columns=[id_col, target_col])
   y = df[target_col]
   print("Missing values per column before processing:")
   print(X.isna().sum())
   if X.isna().any().any():
       print("Missing values detected and will be handled via imputation in pipeline.")
   categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
   numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
   print(f"Numerical columns: {numerical_cols}")
   print(f"Categorical columns: {categorical_cols}")
   numeric_transformer = Pipeline(
       steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
   )
   categorical_transformer = Pipeline(
       steps=[
           ("imputer", SimpleImputer(strategy="most_frequent")),
           ("onehot", OneHotEncoder(handle_unknown="ignore")),
       ]
   )
   preprocessor = ColumnTransformer(
       transformers=[
           ("num", numeric_transformer, numerical_cols),
           ("cat", categorical_transformer, categorical_cols),
       ]
   )
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, stratify=y, test_size=0.2, random_state=random_state
   )
   print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
   models = {
       "logistic_regression": {
           "pipeline": Pipeline(
               steps=[
                   ("preprocessor", preprocessor),
                   ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
               ]
           ),
           "params": {"clf__C": [0.01, 0.1, 1, 10]},
       },
       "random_forest": {
           "pipeline": Pipeline(
               steps=[
                   ("preprocessor", preprocessor),
                   ("clf", RandomForestClassifier(class_weight="balanced", random_state=random_state)),
               ]
           ),
           "params": {
               "clf__n_estimators": [100, 200],
               "clf__max_depth": [None, 10, 20],
           },
       },
       "xgboost": {
           "pipeline": Pipeline(
               steps=[
                   ("preprocessor", preprocessor),
                   (
                       "clf",
                       XGBClassifier(
                           scale_pos_weight=2,
                           eval_metric="logloss",
                           random_state=random_state,
                       ),
                   ),
               ]
           ),
           "params": {
               "clf__n_estimators": [100, 200],
               "clf__max_depth": [3, 5],
               "clf__learning_rate": [0.01, 0.1, 0.3],
           },
       },
   }


   mlflow.set_experiment(mlflow_experiment_name)


   best_model_name = None
   best_model = None
   best_f1_score = 0


   for name, config in models.items():
       print(f"\nTraining model: {name}")


       with mlflow.start_run(run_name=name):
           grid = GridSearchCV(
               estimator=config["pipeline"],
               param_grid=config["params"],
               scoring="f1",
               cv=5,
               n_jobs=-1,
           )
           grid.fit(X_train, y_train)


           best_estimator = grid.best_estimator_


           y_pred = best_estimator.predict(X_test)
           if hasattr(best_estimator, "predict_proba"):
               y_proba = best_estimator.predict_proba(X_test)[:, 1]
           else:
               y_proba = best_estimator.decision_function(X_test)
               y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())


           acc = accuracy_score(y_test, y_pred)
           prec = precision_score(y_test, y_pred)
           rec = recall_score(y_test, y_pred)
           f1 = f1_score(y_test, y_pred)
           roc_auc = roc_auc_score(y_test, y_proba)


           print("Best Params:", grid.best_params_)
           print("Classification Report:\n", classification_report(y_test, y_pred))
           print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
           print(f"Accuracy: {acc:.4f}")
           print(f"Precision: {prec:.4f}")
           print(f"Recall: {rec:.4f}")
           print(f"F1 Score: {f1:.4f}")
           print(f"ROC AUC: {roc_auc:.4f}")


           mlflow.log_params(grid.best_params_)
           mlflow.log_metric("accuracy", acc)
           mlflow.log_metric("precision", prec)
           mlflow.log_metric("recall", rec)
           mlflow.log_metric("f1_score", f1)
           mlflow.log_metric("roc_auc", roc_auc)


           model_path = os.path.join(model_dir, f"{name}_model.pkl")
           joblib.dump(best_estimator, model_path)
           mlflow.log_artifact(model_path, artifact_path="models")


           fpr, tpr, _ = roc_curve(y_test, y_proba)
           plt.figure()
           plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
           plt.plot([0, 1], [0, 1], "k--")
           plt.xlabel("False Positive Rate")
           plt.ylabel("True Positive Rate")
           plt.title(f"ROC Curve - {name}")
           plt.legend()
           plt.grid(True)
           roc_path = os.path.join(report_dir, f"{name}_roc_curve.png")
           plt.savefig(roc_path)
           plt.close()
           mlflow.log_artifact(roc_path, artifact_path="reports")


           if f1 > best_f1_score:
               best_f1_score = f1
               best_model = best_estimator
               best_model_name = name


   if best_model_name is not None:
       print(f"\nBest model: {best_model_name} with F1 score: {best_f1_score:.4f}")


       signature = infer_signature(X_test, best_model.predict(X_test))


       mlflow.sklearn.log_model(
           sk_model=best_model,
           artifact_path="best_model",
           registered_model_name=f"{best_model_name}_final",
           signature=signature,
           input_example=X_test.head(5)
       )
   else:
       print("No model was trained successfully.")


   return best_model_name, best_model, best_f1_score



