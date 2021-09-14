from typing import Any
from layer import Dataset


def build_feature(sdf: Dataset("titanic")) -> Any:
    df = sdf.to_pandas()
    fare = df[['PASSENGERID','FARE']]
    return fare
