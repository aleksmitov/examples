# What is Layer?
Layer is a [Declarative MLOps Platform](https://layer.co/) that empowers Data Science teams to implement end-to-end machine learning with minimal effort. Layer is a managed solution and can be used without the hassle involved in configuring and setting up servers. 


# Getting started
This project illustrates how to train a natural language process model in Layer using pre-trained word embeddings.
Let's start by installing Layer:
```
pip install layer-sdk
```

Clone this  natural language processing with Layer project:
```
layer clone https://github.com/layerml/examples.git
```
Change into the word embeddings folder run the project:
```
cd examples/nlp-word-embeddings
layer start

```
## File Structure

```yaml
.
|____.layer
| |____project.yaml
|____models
| |____model
| | |____requirements.txt
| | |____model.py
| | |____nlp_model.yaml
|____README.md
|____data
| |____spam_data
| | |____spam_features.yaml
| |____sms_featureset
| | |____spam_features.yaml
| | |____message
| | | |____requirements.txt
| | | |____feature.py
| | |____is_spam
| | | |____requirements.txt
| | | |____feature.py
|____notebooks
| |____spam.ipynb
```
