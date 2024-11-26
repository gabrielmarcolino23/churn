import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Carregar o modelo salvo
model = joblib.load('best_random_forest_model.pkl')

# 2. Ler a tabela de clientes ativos
clientes_ativos = pd.read_csv('DataSet-treinamento/basetest.csv')

# 3. Preparar os dados
# Remover as colunas desnecessárias
variaveis_nao_utilizadas = ['companyId', 'createdAt', 'deletedAt', 'plan', 'churn']
X = clientes_ativos.drop(columns=variaveis_nao_utilizadas, errors='ignore')

# Tratar colunas que contêm valores numéricos mal formatados
for column in X.select_dtypes(include=['object']).columns:
    X[column] = (
        X[column]
        .str.replace('.', '', regex=False)  # Remove pontos (separadores de milhares)
        .str.replace(',', '.', regex=False)  # Substitui vírgulas por pontos (separador decimal)
        .astype(float)  # Converte para float
    )
    
# Manter as colunas necessárias para o output
y_true = clientes_ativos['churn']  # Churn verdadeiro
company_ids = clientes_ativos['companyId']  # IDs das empresas

# 4. Fazer predições
try:
    y_pred = model.predict(X)
except ValueError as e:
    print(f"Erro ao fazer predições: {e}")
    exit()

# 5. Adicionar a predição ao dataframe
clientes_ativos['churn_prediction'] = y_pred

# 6. Comparar e exibir resultados
# Exibir os IDs, churn verdadeiro e predito
resultado = clientes_ativos[['companyId', 'churn', 'churn_prediction']]
print("Resultados:")
print(resultado.head())

# 7. Análise de desempenho
cm = confusion_matrix(y_true, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não Churn', 'Churn'])
cmd.plot(cmap='Blues')
plt.title("Matriz de Confusão - Modelo de Predição")
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred))

# 8. Análise das variáveis de predição padrão
feature_importances = model.feature_importances_
variaveis = X.columns
importances_df = pd.DataFrame({'Variável': variaveis, 'Importância': feature_importances})
importances_df = importances_df.sort_values(by='Importância', ascending=False)

print("\nVariáveis Mais Importantes no Modelo:")
print(importances_df.head())

# Plotar as importâncias
plt.figure(figsize=(10, 6))
plt.bar(importances_df['Variável'], importances_df['Importância'])
plt.xticks(rotation=90)
plt.title("Importância das Variáveis no Modelo")
plt.xlabel("Variáveis")
plt.ylabel("Importância")
plt.show()
