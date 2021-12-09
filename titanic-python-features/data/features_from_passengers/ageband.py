import pandas as pd
from typing import Any
from layer import Context, Dataset


def clean_age(data):
    age = data[0]
    pclass = data[1]
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


def build_feature(context: Context, sdf: Dataset("titanic")) -> Any:
    df = sdf.to_pandas()
    feature_data = df[["PASSENGERID", "AGE","PCLASS"]]
    age = feature_data[['AGE', 'PCLASS']].apply(clean_age, axis=1)
    feature_data['AGE'] = pd.DataFrame(age)
    return feature_data[["PASSENGERID","AGE"]]
