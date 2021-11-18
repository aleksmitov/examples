# Spam Detection Project Example

An NLP example with `nltk` library to predict the spam SMS messages. In this project, we are going to use Python Features to remove stop words and to lemmatize messages. Also, we are going to load an ML model from the Layer Model Catalog to create training data for the `spam_detection` model.

## What we are going to learn?

- Extract advanced features from our data with Python Features utilizing `nltk` and `scikit` libraries.
- Use a model to create a training data for another model
- Experimentation tracking with logging metrics: `f1_score`, `accuracy` and `mean_scores`

## Installation & Running

To check out the Layer Spam Detection example, run:

```bash
layer clone https://github.com/layerml/examples.git
cd examples/spam-detection
```

To run the project:

```bash
layer start
```

## File Structure

```yaml
.
|____.layer
| |____project.yaml               # Project configuration file
|____models
| |____vectorizer
| | |____requirements.txt         # Environment config file
| | |____model.py                 # Source code of the `Vectorizer` model
| | |____vectorizer_model.yaml    # Training directives of our model
| |____spam_detection
| | |____spam_detection_model.yaml  # Training directives of our model
| | |____requirements.txt           # Environment config file
| | |____model.py                   # Source code of the `Spam Detection` model
|____README.md
|____data
| |____spam_data
| | |____spam_data.yaml        # Declares where our source `spam_messages` dataset is
| |____sms_featureset
| | |____sms_features.yaml
| | |____message
| | | |____requirements.txt     # Environment config file for the `is_spam` feature
| | | |____feature.py           # Source code of the `message` feature. We remove stop words and lemmatize messages.
| | |____is_spam
| | | |____requirements.txt    # Environment config file for the `is_spam` feature
| | | |____feature.py         # Source code of the `is_spam` feature. We do basic labelencoding.
|____notebooks
| |____spam_detection.ipynb   # Notebook showing how to use the generated entities in a Jupyter Notebook


```

