
# Enzyme classification project
***
Given 31 features of an observation, predict probability that it has enzymes EC1 and EC2. (multi label binary classification)

This project is based on Kaggle challenge available [here](https://www.kaggle.com/competitions/playground-series-s3e18/overview).

###  📚 **Data preprocessing and model choice.**
* All features were numerical, used a StandardScaler.
* Feature engineering attempts did not yield a better result. (decrease result by 0.003)
* Inspected box plots of each feature and removed 2.5% of samples. (LocalOutlierFactor, capping did not yield good results) (this increased result by 0.004)
* Used Microsoft's FLAML AutoML library to check feature space of 4 models: XGBoost, Random Forest, L1 Logistic Regression, L2 Logistic Regression.
* XGBoost was chosen.

*Working file and training data can be found in "eda" folder.*

### ⭐️ Performance
* Kaggle metric was mean ROC AUC for the 2 labels
  * Training data: 0.6456
  * Test data: 0.6495
  * Kaggle personal submission: 0.6514
  * Kaggle top result: 0.66242
  * Leaderboard: 376/1056

### 🚀 Model deployment
* Deployed on Streamlit, live version can be found [here](https://enzyme-classification.streamlit.app/).
* Given an input file, model can run predictions for both labels and provide a download file.
  * *A sample file is also provided*
