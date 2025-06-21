import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle

MODEL_PATH = r'D:\OneDrive\Área de Trabalho\classificacao_residuos\models\modelo_cnn.h5'
ENCODER_PATH = r'D:\OneDrive\Área de Trabalho\classificacao_residuos\models\label_encoder.pkl'

model = load_model(MODEL_PATH)

with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

def processar_imagem(frame):
    imagem = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
    imagem = imagem.resize((128, 128))
    imagem = np.array(imagem) / 255.0
    imagem = imagem.astype(np.float32)
    imagem = np.expand_dims(imagem, axis=0)
    return imagem

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imagem = processar_imagem(frame)
    pred = model.predict(imagem)
    classe_idx = np.argmax(pred)
    classe = label_encoder.inverse_transform([classe_idx])[0]

    cv2.putText(frame, f'Classe: {classe}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Classificacao de Residuos - Pressione Q para sair', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
