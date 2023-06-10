import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


os.chdir("/Users/kacpergruca/Documents/Studia/UW/2_semestr/Machine learning 1/ML_classification")

df = pd.read_csv('data/input/client_attrition_train.csv')
df_test = pd.read_csv('data/input/client_attrition_test.csv')

def process_dataframe(df, if_train):
    df['customer_sex'] = df['customer_sex'].map({'M': 1, 'F': 0})
    salary_mapping = {
        'below 40K': 1,
        '40-60K': 2,
        '60-80K': 3,
        '80-120K': 4,
        '120K and more': 5
    }
    df['customer_salary_range'] = df['customer_salary_range'].map(salary_mapping)

    education_mapping = {
        'Unknown': np.nan,
        'Uneducated': 1,
        'High School': 2,
        'College': 3,
        'Graduate': 4,
        'Post-Graduate': 5,
        'Doctorate': 6
    }
    df['customer_education'] = df['customer_education'].map(education_mapping)

    df['customer_civil_status'].replace('Unknown', pd.NA, inplace=True)
    df_encoded = pd.get_dummies(df['customer_civil_status'], drop_first=True)
    df_encoded.rename(columns={'Married': 'customer_married', 'Single': 'customer_single'}, inplace=True)

    df = pd.concat([df, df_encoded], axis=1)

    classification_mapping = {
        'Blue': 1,
        'Silver': 2,
        'Gold': 3,
        'Platinum': 4
    }
    df['credit_card_classification'] = df['credit_card_classification'].map(classification_mapping)

    if if_train:
        df['account_status'] = df['account_status'].map({'open': 1, 'closed': 0})

    df = df.drop(['customer_civil_status','customer_id'], axis=1)
    
    return df

def scaling(df, x_col):
    scaler = MinMaxScaler()
    df.loc[:,x_col] = pd.DataFrame(scaler.fit_transform(df.loc[:,x_col]), columns = df.loc[:,x_col].columns)
    df.loc[:,x_col].head()

    imputer = KNNImputer(n_neighbors=5)
    df.loc[:,x_col] = pd.DataFrame(imputer.fit_transform(df.loc[:,x_col]),columns = df.loc[:,x_col].columns)

    return df

x_col = ['customer_age', 'customer_sex',
       'customer_number_of_dependents', 'customer_education',
       'customer_salary_range',
       'customer_relationship_length', 'customer_available_credit_limit',
       'credit_card_classification', 'total_products', 'period_inactive',
       'contacts_in_last_year', 'credit_card_debt_balance',
       'remaining_credit_limit', 'transaction_amount_ratio',
       'total_transaction_amount', 'total_transaction_count',
       'transaction_count_ratio', 'average_utilization',
       'customer_married', 'customer_single']

y_col = ['account_status']

mi_score = ['period_inactive',
            'contacts_in_last_year',
            'credit_card_debt_balance',
            'remaining_credit_limit',
            'transaction_amount_ratio',
            'total_transaction_amount',
            'total_transaction_count',
            'transaction_count_ratio',
            'average_utilization']


def main(df, df_test, x_col, mi_score, y_col):
    df = process_dataframe(df, if_train=True)
    df_test = process_dataframe(df_test, if_train=False)   

    df = scaling(df, x_col)
    df_test = scaling(df_test, x_col)

    base_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=4)
    bagging_model = BaggingClassifier(base_model, random_state=42)
    bagging_model.fit(df.loc[:, mi_score], df.loc[:, y_col])

    df_test['Prediction_account_status'] = bagging_model.predict(df_test.loc[:, mi_score])

    df_test.to_csv('data/output/Prediction_account_status.csv', index=False)

    print(df_test)

main(df, df_test, x_col, mi_score, y_col)
