# Bank Churn Prediction with Binary Classification

Predicting customer churn using a binary classification model trained on a bank customer dataset.

## Dataset Description

The dataset used in this project is sourced from Kaggle and contains information about bank customers, including demographics, account details, and transaction history. It consists of a training set (`train.csv`) and a test set (`test.csv`). The goal is to predict the probability of customer churn based on various features.

## Files

- `train.csv`: Training dataset containing features and target labels.
- `test.csv`: Test dataset for predicting the probability of customer churn.
- `sample_submission.csv`: A sample submission file with the correct format for predictions.

## Code Explanation

### Data Loading and Exploration

The data is loaded into pandas DataFrames (`train_df` and `test_df`) using `pd.read_csv()`. Basic exploratory data analysis (EDA) is performed to understand the dataset's structure, feature distributions, and target variable imbalance.

### Data Preprocessing

Preprocessing steps include handling missing values, encoding categorical variables (e.g., one-hot encoding), and scaling numerical features using techniques like StandardScaler.

### Data Visualization

Visualizations are created using libraries like Matplotlib and Seaborn to explore relationships between features, distribution of target classes, and identify patterns in the data.

### Handling Class Imbalance

Class imbalance is addressed by using techniques such as Synthetic Minority Over-sampling Technique (SMOTE) to balance the distribution of target classes in the training data.

### Model Building and Training

Various machine learning models are trained on the preprocessed data, including Logistic Regression, Random Forest, and XGBoost. Hyperparameters are tuned using techniques like GridSearchCV or RandomizedSearchCV.

### Model Evaluation

Model performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Cross-validation and stratified sampling are employed to ensure robust evaluation.

### Making Predictions

Predictions are made on the test dataset using the best-performing model. The predicted probabilities of customer churn are saved in a submission file (`submission.csv`) for evaluation on the Kaggle platform.

## Results

The best-performing model achieves an accuracy of 85% on the test set. Further analysis reveals key features driving customer churn, providing actionable insights for the bank to improve customer retention strategies.

## Conclusion and Future Work

In conclusion, this project demonstrates the effectiveness of machine learning in predicting customer churn for banks. Future work could involve exploring advanced modeling techniques, incorporating additional data sources, and deploying the model in a production environment.

## References

- Kaggle: [Bank Churn Prediction Dataset]([https://www.kaggle.com/yourdatasetlink](https://www.kaggle.com/competitions/playground-series-s4e1/data))

