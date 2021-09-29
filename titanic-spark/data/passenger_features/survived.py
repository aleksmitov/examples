from typing import Any

from layer import Dataset


def build_feature(passengers: Dataset("titanic_dataset")) -> Any:
    passengers_df = passengers.to_spark()

    # Convert Spark DataFrame into Pandas DataFrame
    return passengers_df.select("PassengerId", "Survived").toPandas()
