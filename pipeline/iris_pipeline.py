from typing_extensions import Annotated  # or `from typing import Annotated on Python 3.9+
from typing import Tuple
import pandas as pd
from zenml import pipeline, step, ArtifactConfig, log_model_metadata, get_step_context
from zenml.materializers import pandas_materializer 
import joblib
import logging
from sklearn.base import ClassifierMixin

# Configure logging
logging.basicConfig(level=logging.INFO)

@step
def iris_data_loader() -> (Annotated[pd.DataFrame, "data"]):
    from sklearn.datasets import load_iris
    """Load the Iris dataset as a Pandas DataFrame."""
    logging.info("Loading Iris dataset...")
    iris = load_iris(as_frame=True)
    logging.info("Done loading")
    data = iris.frame
    return data  # Returns a DataFrame containing both features and target

@step
def data_splitter(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    
    from sklearn.model_selection import train_test_split
    """Split the data into training and testing sets."""
    logging.info("Splitting data into train and test sets...")
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    logging.info(f"Training set size: {X_train.shape[0]} samples")
    logging.info(f"Testing set size: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

@step()
def svc_trainer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    gamma: float = 0.001,
    random_state: int = 42
) -> Annotated[ClassifierMixin, ArtifactConfig(name="sklearn_classifier", is_model_artifact=True)]:
    
    from sklearn.base import ClassifierMixin
    from sklearn.svm import SVC
    """Train an SVM classifier using the training data."""
    logging.info("Training SVM classifier...")
    model = SVC(gamma=gamma, random_state=random_state)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    logging.info(f"Training accuracy: {train_acc:.4f}")
    
    step_context = get_step_context()

    log_model_metadata(
            metadata={
                "evaluation_metrics":{
                    "accuracy": train_acc
                    }
                }
            )
    
    return model


@pipeline()
def training_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
    gamma: float = 0.001
):
    """Pipeline to load data, train an SVM classifier, and save the model."""
    # Step 1: Load data
    data = iris_data_loader()
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = data_splitter(
        data=data,
        test_size=test_size,
        random_state=random_state
    )
    
    # Step 3: Train SVM
    svc_model = svc_trainer(
        X_train=X_train,
        y_train=y_train,
        gamma=gamma,
        random_state=random_state
    )
    
    return svc_model 

if __name__ == "__main__":
    # Execute the pipeline with default parameters
    training_pipeline(test_size = 0.1, random_state=43, gamma = 0.002)


