from typing import Any

from pyspark.sql.functions import avg, col, when

from layer import Dataset


def build_feature(passengers: Dataset("titanic_dataset")) -> Any:
    passengers_df = passengers.to_spark()

    avg_age_df = passengers_df \
        .filter(col("Age").isNotNull()) \
        .groupBy("PClass", "Sex") \
        .agg(avg("Age").alias("AvgAge")) \
        .select("PClass", "Sex", "AvgAge")

    passengers_age_df = passengers_df.join(avg_age_df, ["PClass", "Sex"], "left") \
        .select("PASSENGERID",
                when(col("Age").isNull(), col("AvgAge")).otherwise(col("Age")).alias("Age")
                )

    passengers_age_band_df = passengers_age_df.withColumn("AGE_BAND", when(col("Age") <= 16, 0)
                                                          .when((col("Age") > 16) & (col("Age") <= 32), 1)
                                                          .when((col("Age") > 32) & (col("Age") <= 48), 2)
                                                          .when((col("Age") > 48) & (col("Age") <= 64), 3)
                                                          .otherwise(4))

    return passengers_age_band_df.select("PASSENGERID", "AGE_BAND")
