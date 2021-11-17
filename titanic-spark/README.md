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
| |____project.yaml # Project configuration file
|____models
| |____survival_model
| | |____requirements.txt # Environment config file
| | |____model.py  # Source code of the `Survival` model
| | |____survival_model_training_spark.yaml # Training directives of our model
|____README.md
|____data
| |____titanic_data
| | |____titanic_dataset.yaml  # Declares where our source `titanic` dataset is
| |____passenger_features     # feature definitions
| | |____requirements.txt     # Environment config file
| | |____passenger_features_spark.yaml # Declares the metadata of the features
| | |____sex.py              # Sex of the passenger
| | |____ageband.py         # Age Band of the passenger
| | |____is_alone.py        # Is Passenger travelling alone
| | |____survived.py        # Survived or not
| | |____embarked.py        # Embarked or not
| | |____fareband.py        # Fare Band of the passenger
| | |____title.py           # Title of the passenger
|____notebooks
| |____titanic_spark.ipynb # File showing how to use the generated entities in a notebook

```

