"""New Project Example

This file demonstrates how we can develop and train our model by using the
`features` we've developed earlier. Every ML model project
should have a definition file like this one.

"""
from typing import Any
from layer import Train, Dataset
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


def train_model(train: Train, ratings: Dataset("movie_ratings"),) -> Any:
    """Model train function

    This function is a reserved function and will be called by Layer
    when we want this model to be trained along with the parameters.
    Just like the `features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.

    Args:
        train (layer.Train): Represents the current train of the model, passed by
            Layer when the training of the model starts.
        pf (spark.DataFrame): Layer will return all features inside the
            `features` featureset as a spark.DataFrame automatically
            joining them by primary keys, described in the dataset.yml

    Returns:
       model: Trained model object

    """
    data = ratings.to_spark()
    # Split the data into a training and testing set
    training_size = 0.7
    random_state = 0
    test_size = 0.3
    train.log_parameters({
        "training_size": training_size,
        "random_state": random_state,
        "test_size": test_size
    })
    training, testing = data.randomSplit([training_size, test_size], seed=random_state)
    # Recommendation model using ALS on the training data
    # model parameters
    maxIter = 5
    regParam = 0.01
    userCol = "USERID"
    itemCol = "MOVIEID"
    ratingCol = "RATING"
    coldStartStrategy = "drop"
    train.log_parameters({
        "maxIter": maxIter,
        "regParam": regParam,
        "userCol": userCol,
        "itemCol": itemCol,
        "ratingCol": ratingCol,
        "coldStartStrategy": coldStartStrategy
    })
    # https://spark.apache.org/docs/3.0.0/ml-collaborative-filtering.html
    # According to https://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS
    # ALS attempts to estimate the ratings matrix R as the product of two lower-rank matrices, X and Y, i.e. X * Yt = R.
    # Typically these approximations are called ‘factor’ matrices. The general approach is iterative. During each iteration,
    # one of the factor matrices is held constant, while the other is solved for using least squares. The newly-solved factor
    # matrix is then held constant while solving for the other factor matrix.
    als = ALS(maxIter=maxIter, regParam=regParam, userCol=userCol, itemCol=itemCol, ratingCol=ratingCol,
              coldStartStrategy=coldStartStrategy  )
    model = als.fit(training)
    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(testing)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol=ratingCol, predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    train.log_metric("RMSE", rmse)
    return model

