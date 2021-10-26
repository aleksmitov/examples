import layer
from typing import Any


def build_feature() -> Any:
    # Fetch "passenger_features_spark" featureset into Spark DataFrame
    titanic_features = layer.get_featureset("passenger_features_spark").to_spark()
    titanic_features = titanic_features.drop("Sex")

    # Building a new featureset
    return titanic_features
