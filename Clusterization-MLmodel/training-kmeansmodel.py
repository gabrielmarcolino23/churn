import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # Importar explicitamente
from sklearn.impute import IterativeImputer

# Carregar a base de dados completa
dataset = pd.read_csv('DataSet-treinamento/model_dataset.csv')  # Substitua pelo caminho correto

# Filtrar os dados onde churn = 1
churn_data = dataset[dataset['churn'] == 1].copy()

# Selecionar todas as features exceto 'churn'
X = churn_data.drop(columns=['churn'])

# Escalonar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar IterativeImputer para imputar valores ausentes
imputer = IterativeImputer(random_state=42)
X_imputed = imputer.fit_transform(X_scaled)

# Aplicar GMM com 3 clusters (baixo, médio, alto risco)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_imputed)

# Prever os clusters
churn_data['risk_cluster'] = gmm.predict(X_imputed)

# Mapear os clusters para 'Baixo', 'Médio', 'Alto' com base nos centros
cluster_centers = gmm.means_.sum(axis=1)
risk_mapping = {cluster: risk for cluster, risk in zip(cluster_centers.argsort(), ['Baixo', 'Médio', 'Alto'])}
churn_data['risk_level'] = churn_data['risk_cluster'].map(risk_mapping)

# Exibir os resultados
print("\nExemplo de Risco de Churn:")
print(churn_data[['risk_cluster', 'risk_level']].head())

# Salvar os resultados
churn_data.to_csv('clientes_churnados_com_risco.csv', index=False)
print("\nBase de dados salva como 'clientes_churnados_com_risco.csv'.")
