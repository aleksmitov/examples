import layer
from typing import Any


def build_feature() -> Any:
    # Convert Layer featuresets into Spark
    titanic_features = layer.get_featureset("passenger_features_spark").to_spark()

    # Building a new featureset
    return titanic_features
