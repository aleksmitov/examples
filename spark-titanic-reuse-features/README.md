# Titanic Survival Example

A classification example with `Spark ML` for predicting the survivals of the Titanic passengers. We will be using the famous [Kaggle Titanic](https://www.kaggle.com/c/titanic/data?select=train.csv) dataset.

## What we are going to learn?

- How to use existing features to create a new `featureset` in Spark. The project depends on the `passenger_features_spark` featureset. 
Feature Store: We are going to use PySpark interface to build the `passenger` features 
- Utilize `SparkSession` from Layer `Context` for Spark SQL queries (i.e. `title` feature)
- Convert Spark DataFrames into Pandas DataFrames as the way Layer stores the features
- Load `passenger` features and use it to train our `survival` model
- Experimentation tracking with
    - logging `BinaryClassificationEvaluator` metric

## Install and run

To check out the Layer Titanic Survival example, run:

```bash
layer clone https://github.com/layerml/examples
cd examples/spark-titanic-reuse-features
```

To run the project:

```bash
layer start
```

## File Structure

```yaml
..
|____.layer
| |____project.yaml
|____models
| |____survival_model
| | |____model.yaml
| | |____requirements.txt
| | |____model.py
|____README.md
|____data
| |____titanic_data
| | |____dataset.yaml
| |____passenger_features
| | |____requirements.txt
| | |____features.py
| | |____dataset.yaml

```

