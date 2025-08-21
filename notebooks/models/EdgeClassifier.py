from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as XGB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib
import os


class EdgeClassifier(ABC):
    """
    Abstract base class for edge classifiers.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.edge_type = dataloader.edge_type
        self.save_dir = dataloader.save_dir
        # Define categorical features for one-hot encoding
        categorical_features = [
            'src_status', 'src_type', 'tgt_status',
        ] if self.edge_type == 'vr_graph' else [
            'src_status', 'tgt_status'
        ]
        self.classifier = None  # Placeholder for the classifier
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_features)
            ],
            remainder='passthrough'  # Keep other features as they are
        )
        self.pipeline = None  # Placeholder for the pipeline

    def classify_edges(self):
        """
        Classify edges based on the provided features.

        Returns:
            A list of classifications for the edges.
        """
        if self.pipeline is None:
            raise NotImplementedError("Pipeline is not defined.")

        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_data()

        # Try to load existing model
        if self.dataloader.load_dir is not None:
            suffix = self.edge_type.split('_')[0]
            classifier_name = self.classifier.__class__.__name__.lower()
            model_path = f'{self.dataloader.load_dir}/{classifier_name}_{suffix}.pkl'
            if os.path.exists(model_path):
                print(f"Loading existing model from {model_path}")
                self.pipeline = joblib.load(model_path)
                return self.pipeline.predict_proba(X_val)[:, 1], self.pipeline.predict_proba(X_test)[:, 1]

        # Fit the model
        print(
            f"Training {self.classifier.__class__.__name__} for edge type: {self.edge_type}")
        self.pipeline.fit(X_train, y_train.squeeze())

        # Save the model
        suffix = self.edge_type.split('_')[0]
        classifier_name = self.classifier.__class__.__name__.lower()
        with open(f'{self.save_dir}/{classifier_name}_{suffix}.pkl', 'wb') as f:
            joblib.dump(self.pipeline, f)

        # Evaluate the model
        print("Training accuracy:", self.pipeline.score(X_train, y_train))
        print("Validation accuracy:", self.pipeline.score(X_val, y_val))

        # Predict on validation set
        y_val_pred = self.pipeline.predict(X_val)
        y_val_proba = self.pipeline.predict_proba(X_val)[:, 1] if hasattr(
            self.pipeline, "predict_proba") else None

        print("Validation Metrics:")
        print("Accuracy:", accuracy_score(y_val, y_val_pred))
        print("Precision:", precision_score(
            y_val, y_val_pred, zero_division=0))
        print("Recall:", recall_score(y_val, y_val_pred, zero_division=0))
        print("F1 Score:", f1_score(y_val, y_val_pred, zero_division=0))
        if y_val_proba is not None:
            print("ROC AUC:", roc_auc_score(y_val, y_val_proba))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
        print("Classification Report:\n", classification_report(
            y_val, y_val_pred, zero_division=0))

        # Evaluate predictions on train, val, and test sets
        # Training set
        y_train_pred = self.pipeline.predict(X_train)
        print(f"{classifier_name} - Training set classification report:")
        print(classification_report(y_train, y_train_pred))

        # Validation set
        y_val_pred = self.pipeline.predict(X_val)
        print(f"{classifier_name} - Validation set classification report:")
        print(classification_report(y_val, y_val_pred))

        # Test set
        y_test_pred = self.pipeline.predict(X_test)
        print(f"{classifier_name} - Test set classification report:")
        print(classification_report(y_test, y_test_pred))

        return y_val_proba, self.pipeline.predict_proba(X_test)[:, 1]

    def get_data(self):
        """
        Get data for edge classification.

        Returns:
            Tuple of training, validation, and test sets.
        """
        (X_train, y_train), (X_val, y_val), (X_test,
                                             y_test) = self.dataloader.load_data()
        return X_train, y_train, X_val, y_val, X_test, y_test


class RFClassifier(EdgeClassifier):
    """
    Random Forest Classifier for edge classification tasks.
    Inherits from EdgeClassifier and implements the classify_edges method.
    """
    RANDOM_STATE = 42  # For reproducibility
    N_ESTIMATORS = 100  # Default number of trees in Random Forest

    def __init__(self, dataloader, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS):
        super().__init__(dataloader)
        # Initialize the Random Forest classifier
        self.classifier = RandomForestClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
        )
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.classifier)
        ])


class XGBClassifier(EdgeClassifier):
    """XGBoost Classifier for edge classification tasks.
    Inherits from EdgeClassifier and implements the classify_edges method.
    """
    RANDOM_STATE = 42  # For reproducibility
    N_ESTIMATORS = 100  # Default number of trees in Random Forest

    def __init__(self, dataloader, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS):
        super().__init__(dataloader)
        # Initialize the XGBoost classifier
        self.classifier = XGB(
            random_state=random_state,
            n_estimators=n_estimators,
            eval_metric='logloss'
        )
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.classifier)
        ])
