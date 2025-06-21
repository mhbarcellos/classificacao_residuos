import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

DATA_PATH = r'D:\OneDrive\Área de Trabalho\classificacao_residuos\data'
MODEL_PATH = r'D:\OneDrive\Área de Trabalho\classificacao_residuos\models'

X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
X_val = np.load(os.path.join(DATA_PATH, 'X_val.npy'))
y_val = np.load(os.path.join(DATA_PATH, 'y_val.npy'))
X_test = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_flat, y_train)

y_pred = model.predict(X_test_flat)

acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc}")

print("Matriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Random Forest')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

with open(os.path.join(MODEL_PATH, 'modelo_randomforest.pkl'), 'wb') as f:
    pickle.dump(model, f)

print("Modelo Random Forest salvo em models/modelo_randomforest.pkl")

with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
labels = label_encoder.classes_

def analisar_matriz_confusao(cm, labels):
    acertos = np.diag(cm)
    total_por_classe = cm.sum(axis=1)
    taxas_acerto = acertos / total_por_classe

    print("\nAnálise da Matriz de Confusão:")
    for idx, taxa in enumerate(taxas_acerto):
        classe = labels[idx]
        percentual = taxa * 100
        print(f"- {classe}: {percentual:.2f}% de acerto")

    pior_idx = np.argmin(taxas_acerto)
    melhor_idx = np.argmax(taxas_acerto)

    print(f"\nClasse com melhor desempenho: {labels[melhor_idx]} ({taxas_acerto[melhor_idx]*100:.2f}%)")
    print(f"Classe com pior desempenho: {labels[pior_idx]} ({taxas_acerto[pior_idx]*100:.2f}%)")

analisar_matriz_confusao(cm, labels)
