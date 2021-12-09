import pandas as pd
from typing import Any
from layer import Context, RawDataset


def clean_embark(embark):
    result = 0
    if embark == "S":
        result = 0
    elif embark == "C":
        result = 1
    elif embark == "Q":
        result == 2
    else:
        result = -1
    return result


def build_feature(context: Context, sdf: RawDataset("titanic")) -> Any:
    df = sdf.to_pandas()
    feature_data = df[["PASSENGERID", "EMBARKED"]]
    embark = feature_data['EMBARKED'].apply(clean_embark)
    feature_data['EMBARKED'] = pd.DataFrame(embark)

    return feature_data[["PASSENGERID","EMBARKED"]]

