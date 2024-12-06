# Credit-Card-Fraud-detection
This project uses Apache Spark and machine learning to detect fraudulent credit card transactions in imbalanced datasets. It handles class imbalance via oversampling and undersampling, trains models like Logistic Regression, Random Forest, and Gradient Boosting, and evaluates them on metrics like accuracy, precision, recall, and F1 score.

## Project Overview
The script performs the following tasks:

1. Data Loading: Reads the credit card transaction data from a CSV file.
2. Data Inspection: Analyzes the schema and handles missing values.
3. Class Distribution Analysis: Examines the distribution of fraudulent vs. non-fraudulent transactions.
4. Class Imbalance Handling:
5. Oversampling: Increases the number of fraudulent transactions to balance the dataset.
6. Undersampling: Reduces the number of non-fraudulent transactions to balance the dataset.
7. Feature Engineering: Transforms features into a suitable format for machine learning models.
8. Model Training and Evaluation: Trains and evaluates models using Logistic Regression, Random Forest, Naive Bayes, and Gradient Boosting.
9. Assesses model performance based on accuracy, precision, recall, and F1 score.
10. Results Visualization: Presents model performance metrics in tables and plots.

## Getting Started
Prerequisites
Apache Spark (with PySpark)
Python 3.x
Required Python libraries: pyspark, matplotlib, tabulate

## Link to Dataset
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Installation
1. Clone the repository:

   ```shell script
   git clone https://github.com/SanketHingne/credit-card-fraud-detection.git
   ```

2. Navigate to the project directory:
   ```shell script
   cd credit-card-fraud-detection
   ```
3. Install the required Python packages:
   ```shell script
   pip install pyspark matplotlib tabulate
   ```
## Running the Script
Ensure your data file (creditcard.csv) is located correctly in the script.
Run the script:
```shell script
spark-submit credit_card_fraud_detection.py
```
## Results
1. The script will output the performance metrics for each model to the console.
2. Visualizations of model performance will be generated and saved in the project directory.

## File Structure
credit_card_fraud_detection.py: The main script for credit card fraud detection.

## Acknowledgements
1. Apache Spark
2. PySpark Documentation
3. Matplotlib Documentation
