# Titanic Survival Example

A classification example with `sklearn.RandomForestClassifier` for predicting the survivals of the Titanic passengers. We will be using the famous [Kaggle Titanic](https://www.kaggle.com/c/titanic/data?select=train.csv) dataset.

## What we are going to learn?

- Feature Store: We are going to use SQL queries to build the `passenger` features.
- Load `passenger` features and use it to train our `survival` model
- Experimentation tracking with
    - logging `accuracy` metric
    - logging `n_estimators` parameter

## Installation & Running

To check out the Layer Titanic Survival example, run:

```bash
layer clone https://github.com/layerml/examples
cd examples/titanic-python-features
```

To run the project:

```bash
layer start
```

## File Structure

```yaml
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
| |____features_from_passengers
| | |____requirements.txt
| | |____sex.py
| | |____ageband.py
| | |____survived.py
| | |____embarked.py
| | |____dataset.yaml
| | |____fareband.py
```

