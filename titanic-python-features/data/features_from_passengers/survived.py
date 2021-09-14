import pandas as pd
from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("titanic")) -> Any:
    df = sdf.to_pandas()
    survived = df[['PASSENGERID','SURVIVED']]
    return survived
