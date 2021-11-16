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
| |____project.yaml
|____models
| |____vectorizer
| | |____requirements.txt
| | |____model.py
| | |____vectorizer_model.yaml
| |____spam_detection
| | |____spam_detection_model.yaml
| | |____requirements.txt
| | |____model.py
|____README.md
|____data
| |____spam_data
| | |____spam_data.yaml
| |____sms_featureset
| | |____sms_features.yaml
| | |____message
| | | |____requirements.txt
| | | |____feature.py
| | |____is_spam
| | | |____requirements.txt
| | | |____feature.py
|____notebooks
| |____spam_detection.ipynb


```

