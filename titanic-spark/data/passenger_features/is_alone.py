from typing import Any

from pyspark.sql.functions import col, when

from layer import Dataset


def build_feature(passengers: Dataset("titanic_dataset")) -> Any:
    passengers_df = passengers.to_spark()
    family_size_df = passengers_df.withColumn("FamilySize", col("SibSp") + col("Parch") + 1)
    is_alone_df = family_size_df.withColumn("IsAlone", when(col("FamilySize") == 1, 1).otherwise(0))

    return is_alone_df.select("PassengerId", "IsAlone").toPandas()
