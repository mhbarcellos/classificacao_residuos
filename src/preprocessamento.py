import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

DATASET_PATH = r'D:\OneDrive\Área de Trabalho\classificacao_residuos\imagens'
SAVE_DATA_PATH = r'D:\OneDrive\Área de Trabalho\classificacao_residuos\data'
SAVE_MODELS_PATH = r'D:\OneDrive\Área de Trabalho\classificacao_residuos\models'

os.makedirs(SAVE_DATA_PATH, exist_ok=True)
os.makedirs(SAVE_MODELS_PATH, exist_ok=True)

IMAGE_SIZE = (128, 128)

print("=== ETAPA 1: Carregando imagens ===")

imagens = []
labels = []

for categoria in os.listdir(DATASET_PATH):
    categoria_path = os.path.join(DATASET_PATH, categoria)
    
    if os.path.isdir(categoria_path):
        print(f"Processando categoria: {categoria}")
        for arquivo in os.listdir(categoria_path):
            caminho_arquivo = os.path.join(categoria_path, arquivo)

            try:
                img = Image.open(caminho_arquivo).convert('RGB')
                img = img.resize(IMAGE_SIZE)
                img_array = np.array(img, dtype=np.float32) / 255.0

                imagens.append(img_array)
                labels.append(categoria)

            except Exception as e:
                print(f"Erro ao processar {caminho_arquivo}: {e}")

print(f"Total de imagens carregadas: {len(imagens)}")

print("=== ETAPA 2: Convertendo para arrays ===")

X = np.array(imagens, dtype=np.float32)
y = np.array(labels)

print(f"Formato das imagens (X): {X.shape}")
print(f"Formato dos rótulos (y): {y.shape}")

print("=== ETAPA 3: Aplicando Label Encoder ===")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

with open(os.path.join(SAVE_MODELS_PATH, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

print("Label Encoder salvo em models/label_encoder.pkl")
print("Classes detectadas:", list(label_encoder.classes_))

print("=== ETAPA 4: Dividindo os dados ===")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

print(f"Treino: {X_train.shape[0]} imagens")
print(f"Validação: {X_val.shape[0]} imagens")
print(f"Teste: {X_test.shape[0]} imagens")

print("=== ETAPA 5: Salvando arquivos numpy ===")

np.save(os.path.join(SAVE_DATA_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(SAVE_DATA_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(SAVE_DATA_PATH, 'X_val.npy'), X_val)
np.save(os.path.join(SAVE_DATA_PATH, 'y_val.npy'), y_val)
np.save(os.path.join(SAVE_DATA_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(SAVE_DATA_PATH, 'y_test.npy'), y_test)

print("Dados numpy salvos na pasta data/")
print("Processamento concluído com sucesso.")
