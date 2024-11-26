import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
from imblearn.over_sampling import SMOTE  # Substituí o undersampling pelo SMOTE

# Carregando os dados
model_data = pd.read_csv('DataSet-treinamento/model_dataset.csv')

# Separando as features (X) e o alvo (y)
X = model_data.drop('churn', axis=1).reset_index(drop=True)
y = model_data['churn'].reset_index(drop=True)

# Configuração da validação cruzada
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# Configurando o SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Instanciando o modelo Random Forest fora do loop
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Avaliação com validação cruzada (focando na métrica accuracy como exemplo)
cv_scores = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='accuracy')

# Treinando o modelo com todo o conjunto balanceado
model.fit(X_balanced, y_balanced)

# Calculando as métricas de classificação no conjunto de treino
y_train_pred = model.predict(X_balanced)
train_report = classification_report(y_balanced, y_train_pred, output_dict=True)

# Salvando o modelo
joblib.dump(model, 'best_random_forest_model.pkl')
print("\nModelo salvo como 'best_random_forest_model.pkl'")
print(f"Acurácia média na validação cruzada: {np.mean(cv_scores):.4f}")
print("\nRelatório de Classificação no Conjunto de Treino:")
print(classification_report(y_balanced, y_train_pred))

# Calculando e exibindo a matriz de confusão no conjunto de treino
cm = confusion_matrix(y_balanced, y_train_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não churn', 'Churn'])
cmd.plot(cmap='Blues')
cmd.ax_.set_title("Matriz de Confusão (Conjunto de Treino)")
cmd.ax_.set_xlabel("Classes Preditas")
cmd.ax_.set_ylabel("Classes Verdadeiras")
plt.show()

# Valores diretos da matriz de confusão
print(f"Matriz de Confusão:\n{cm}")
