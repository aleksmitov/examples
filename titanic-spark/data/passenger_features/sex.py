from typing import Any

from pyspark.sql.functions import col, when

from layer import Dataset


def build_feature(passengers: Dataset("titanic_dataset")) -> Any:
    passengers_df = passengers.to_spark()
    sex_df = passengers_df.withColumn("Sex", when(col("Sex") == "male", 1).otherwise(0))

    return sex_df.select("PassengerId", "Sex").toPandas()
