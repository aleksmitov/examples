from typing import Any

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from layer import Featureset, Train


def train_model(train: Train, pf: Featureset("spark_passenger_features")) -> Any:
    passenger_df = pf.to_spark()

    feat_cols = ['AgeBand', 'EmbarkStatus', 'FareBand', 'IsAlone', 'Sex', 'Title']
    label_col = 'Survived'

    vec_assember = VectorAssembler(inputCols=feat_cols, outputCol='features')
    final_data = vec_assember.transform(passenger_df)

    test_size = 0.2
    training_size = 0.8
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
