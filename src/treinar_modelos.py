import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# Caminhos
DATA_PATH = r'D:\OneDrive\Área de Trabalho\classificacao_residuos\data'
MODEL_PATH = r'D:\OneDrive\Área de Trabalho\classificacao_residuos\models'

# Carregar dados
X_train = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
X_val = np.load(os.path.join(DATA_PATH, 'X_val.npy'))
y_val = np.load(os.path.join(DATA_PATH, 'y_val.npy'))
X_test = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

# Label Encoder
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

print("Label Encoder salvo.")

labels = label_encoder.classes_
num_classes = len(labels)

# 🔥 Modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
X_train_rf = X_train.reshape(len(X_train), -1)
X_val_rf = X_val.reshape(len(X_val), -1)
X_test_rf = X_test.reshape(len(X_test), -1)

rf.fit(X_train_rf, y_train_enc)
y_pred_rf = rf.predict(X_test_rf)

print("\n=== Random Forest ===")
print("Acurácia:", accuracy_score(y_test_enc, y_pred_rf))
print(classification_report(y_test_enc, y_pred_rf))

with open(os.path.join(MODEL_PATH, 'modelo_randomforest.pkl'), 'wb') as f:
    pickle.dump(rf, f)

print("Modelo Random Forest salvo.")

# 🔥 Modelo CNN
cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

cnn.compile(optimizer=Adam(learning_rate=0.0001),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nTreinando CNN...")
history_cnn = cnn.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

cnn.save(os.path.join(MODEL_PATH, 'modelo_cnn.h5'))
np.save(os.path.join(MODEL_PATH, 'history_cnn.npy'), history_cnn.history)

print("Modelo CNN salvo.")

# 🔥 Modelo Transfer Learning
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

tl = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

tl.compile(optimizer=Adam(learning_rate=0.0001),
           loss='sparse_categorical_crossentropy',
           metrics=['accuracy'])

print("\nTreinando Transfer Learning...")
history_tl = tl.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

tl.save(os.path.join(MODEL_PATH, 'modelo_transfer_learning.h5'))
np.save(os.path.join(MODEL_PATH, 'history_transfer_learning.npy'), history_tl.history)

print("Modelo Transfer Learning salvo.")

print("\n=== Processo Finalizado com Sucesso ===")
