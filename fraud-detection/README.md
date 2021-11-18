# Fraud Detection Example

A gradient boosting example with `xgboost` library to reveal suspicious transactions. We will be working with a transaction log dataset from [Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/ntnu-testimon/paysim1).

For experimentation tracking, we will use the area under the precision-recall curve (AUPRC) rather than the conventional area under the receiver operating characteristic (AUROC), since the data is highly skewed.

## What we are going to learn?

- Feature Store: We are going to use SQL queries to build the `transaction` features.
- Load `transaction` features and use it to train the `fraud_detection_model`
- Experimentation tracking
 - logging `auprc` metric
 - logging parameters: `max_depth` and `objective`

## Installation & Running

To check out the Layer Fraud Detection example, run:

```bash
layer clone https://github.com/layerml/examples.git
cd examples/fraud-detection
```

To run the project:

```bash
layer start
```

## File Structure

```yaml
.
|____.layer
| |____project.yaml  # Project configuration file
|____models
| |____fraud_detection_model   
| | |____requirements.txt     # Environment config file
| | |____fraud_detection_model.yaml # Training directives of our model
| | |____model.py    # Source code of the fraud detection model
|____README.md
|____data
| |____transactions # feature definitions
| | |____transaction_dataset.yaml # Declares the metadata of the features 
| |____transaction_features
| | |____new_balance_dest.sql   # New Balance on the destination account
| | |____old_balance_orig.sql   # Old Balance on the originating account
| | |____error_balance_dest.sql # Error Balance on the destination account
| | |____is_fraud.sql           # Our target value
| | |____type.sql               # Type of the transaction feature
| | |____transaction_features.yaml # Declares where our source `transactions` dataset is
| | |____new_balance_orig.sql    # New Balance on the originating account
| | |____error_balance_orig.sql # Error Balance on the originating account
| | |____old_balance_dest.sql   # Old Balance on the destination account
```
