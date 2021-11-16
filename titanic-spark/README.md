# Titanic Survival Example

A classification example with `Spark ML` for predicting the survivals of the Titanic passengers. We will be using the famous [Kaggle Titanic](https://www.kaggle.com/c/titanic/data?select=train.csv) dataset.

## What we are going to learn?

- Feature Store: We are going to use PySpark interface to build the `passenger` features 
- Utilize `SparkSession` from Layer `Context` for Spark SQL queries (i.e. `title` feature)
- Convert Spark DataFrames into Pandas DataFrames as the way Layer stores the features
- Load `passenger` features and use it to train our `survival` model
- Run the model in a Spark cluster, as specified by the `fabric` setting 
- Experimentation tracking with
    - logging `BinaryClassificationEvaluator` metric

## Installation & Running

To check out the Layer Titanic Survival example, run:

```bash
layer clone https://github.com/layerml/examples
cd examples/titanic-spark
```

To run the project:

```bash
layer start
```

## File Structure

```yaml
.
|____.layer
| |____project.yaml
|____models
| |____survival_model
| | |____requirements.txt
| | |____model.py
| | |____survival_model_training_spark.yaml
|____README.md
|____data
| |____titanic_data
| | |____titanic_dataset.yaml
| |____passenger_features
| | |____requirements.txt
| | |____passenger_features_spark.yaml
| | |____sex.py
| | |____ageband.py
| | |____is_alone.py
| | |____survived.py
| | |____embarked.py
| | |____fareband.py
| | |____title.py
|____notebooks
| |____titanic_spark.ipynb

```

