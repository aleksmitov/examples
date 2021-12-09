from typing import Any

from pyspark.sql.functions import col, when

from layer import Context, RawDataset


def build_feature(context: Context, passengers: RawDataset("titanic_dataset")) -> Any:
    passengers_df = passengers.to_spark()
    sex_df = passengers_df.withColumn("SEX", when(col("Sex") == "male", 1).otherwise(0))

    return sex_df.select("PASSENGERID", "SEX")
