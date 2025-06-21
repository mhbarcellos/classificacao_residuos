import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
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
num_classes = len(labels)

base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\nIniciando treinamento...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    callbacks=[early_stop]
)

print("\nTreinamento concluído.\n")

loss, acc = model.evaluate(X_test, y_test)
print(f"\nAcurácia no teste: {acc * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - MobileNetV2')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nSalvando modelo e histórico...\n")

model.save(os.path.join(MODEL_PATH, 'modelo_transfer_learning.h5'))
np.save(os.path.join(MODEL_PATH, 'history_transfer_learning.npy'), history.history)

print("Modelo salvo em: models/modelo_transfer_learning.h5")
print("Histórico salvo em: models/history_transfer_learning.npy")

print("\nProcesso finalizado com sucesso ✅")
