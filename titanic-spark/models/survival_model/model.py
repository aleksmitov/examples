from typing import Any

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from layer import Featureset, Context


def train_model(context: Context, pf: Featureset("passenger_features_spark")) -> Any:
    passenger_df = pf.to_spark()

    feat_cols = ['AGE_BAND', 'EMBARK_STATUS', 'FARE_BAND', 'IS_ALONE', 'SEX', 'TITLE']
    label_col = 'SURVIVED'

    vec_assember = VectorAssembler(inputCols=feat_cols, outputCol='features')
    final_data = vec_assember.transform(passenger_df)

    test_size = 0.2
    training_size = 0.8
    train = context.train()
    train.log_parameter("test_size", test_size)
    seed = 42
    train.log_parameter("seed", seed)
    training, testing = final_data.randomSplit([training_size, test_size], seed=seed)

    lr = LogisticRegression(labelCol=label_col, featuresCol='features')
    survival_model = lr.fit(training)

    predictions = survival_model.transform(testing)
    evaluator = BinaryClassificationEvaluator(labelCol=label_col)

    train.log_metric("BinaryClassificationEvaluator", evaluator.evaluate(predictions))

    return survival_model
