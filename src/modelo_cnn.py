import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
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

with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
labels = label_encoder.classes_

num_classes = len(np.unique(y_train))

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nAcurácia: {acc * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - CNN')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

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

print("\nLegenda dos índices da matriz:")
for idx, classe in enumerate(labels):
    print(f"{idx} → {classe}")

model.save(os.path.join(MODEL_PATH, 'modelo_cnn.h5'))
print("\nModelo CNN salvo em models/modelo_cnn.h5")
