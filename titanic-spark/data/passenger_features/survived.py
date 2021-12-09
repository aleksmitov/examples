from typing import Any

from layer import Context, Dataset


def build_feature(context: Context, passengers: Dataset("titanic_dataset")) -> Any:
    passengers_df = passengers.to_spark()

    return passengers_df.select("PASSENGERID", "SURVIVED")
