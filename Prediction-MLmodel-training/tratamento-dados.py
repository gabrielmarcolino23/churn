# Importing the libraries
import pandas as pd

# Importing the dataset
df = pd.read_csv("C:/Users/danie/OneDrive/Área de Trabalho/Projeto HórusAI/DataSet-treinamento/dataset.csv")

# Target variable
target_variable = df['churn']

# Predictors Variables
predictors_variables = df.drop(['churn'], axis=1)

df.head()

# Remove the columns that are not useful for the model
model_data = predictors_variables.drop(['companyId', 'createdAt','deletedAt', 'plan','dataDif'], axis=1)

# Replace , by . in the columns that are object
for column in model_data.columns:
    if model_data[column].dtype == 'object':
        model_data[column] = model_data[column].str.replace(',', '.').astype(float)

# Adding the target variable to the model_data
model_data['churn'] = target_variable

# # Exporting the data
model_data.to_csv("C:/Users/danie/OneDrive/Área de Trabalho/Projeto HórusAI/DataSet-treinamento/model_dataset.csv", index=False)