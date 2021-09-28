from typing import Any

from pyspark.sql.functions import col, when

from layer import Dataset


def build_feature(passengers: Dataset("titanic_dataset")) -> Any:
    passengers_df = passengers.to_spark()
    embarked_df = passengers_df.withColumn("EmbarkStatus", when(col("Embarked") == "S", 0)
                                           .when(col("Embarked") == "C", 1)
                                           .when(col("Embarked") == "Q", 2)
                                           .otherwise(3))

    return embarked_df.select("PassengerId", "EmbarkStatus").toPandas()
