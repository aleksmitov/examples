from typing import Any

from pyspark.sql.functions import col, when

from layer import Context, Dataset


def build_feature(context: Context, passengers: Dataset("titanic_dataset")) -> Any:
    passengers_df = passengers.to_spark()
    fare_band_df = passengers_df.withColumn("FARE_BAND", when(col("Fare") <= 7.91, 0)
                                            .when((col("Fare") > 7.91) & (col("Fare") <= 14.454), 1)
                                            .when((col("Fare") > 14.454) & (col("Fare") <= 31), 2)
                                            .otherwise(3))

    return fare_band_df.select("PASSENGERID", "FARE_BAND")
