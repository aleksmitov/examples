import pandas as pd
from typing import Any
from layer import Context, RawDataset


def clean_sex(sex):
    result = 0
    if sex == "female":
        result = 0
    elif sex == "male":
        result = 1
    return result


def build_feature(context: Context, sdf: RawDataset("titanic")) -> Any:
    df = sdf.to_pandas()
    feature_data = df[["PASSENGERID", "SEX"]]
    sex = df['SEX'].apply(clean_sex)
    feature_data['SEX'] = pd.DataFrame(sex)

    return feature_data[["PASSENGERID","SEX"]]
