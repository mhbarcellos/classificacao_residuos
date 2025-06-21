
# Classificador de Resíduos Recicláveis

Este projeto tem como objetivo desenvolver um classificador de resíduos utilizando técnicas de **Machine Learning** e **Visão Computacional**. Através de imagens, o modelo é capaz de identificar automaticamente o tipo de resíduo, como papel, plástico, vidro, metal, papelão, entre outros.

## Tecnologias e Modelos Utilizados

-  Random Forest (Scikit-learn)
-  CNN (Keras / TensorFlow)
-  Transfer Learning com MobileNetV2 (Keras / TensorFlow)
-  Interface com Gradio
-  Processamento de imagens com Pillow e NumPy

##  Estrutura do Projeto

```
classificacao_residuos/
├── data/                     → Dados processados (X_train, X_test, etc.)
├── models/                   → Modelos treinados (.h5, .pkl, .npy)
├── src/                      → Scripts Python (treinamento e interface)
│   ├── preprocessamento.py
│   ├── modelo_randomforest.py
│   ├── modelo_cnn.py
│   ├── modelo_transfer_learning.py
│   └── gradio_interface.py
├── README.md                 → Este arquivo
├── requirements.txt          → Dependências
```

##  Como Executar Localmente

1. Clone este repositório:
```
git clone https://github.com/usuario/classificacao_residuos.git
```

2. Instale as dependências:
```
pip install -r requirements.txt
```

3. Execute o aplicativo:
```
python src/gradio_interface.py
```

##  Resultados

| Modelo             | Acurácia |
|--------------------|----------|
| Random Forest      | ~66%     |
| CNN                | ~65%     |
| Transfer Learning  | ~91%     |

O modelo baseado em Transfer Learning (MobileNetV2) apresentou o melhor desempenho, com aproximadamente **91% de acurácia no conjunto de teste.**

## Desenvolvedores

- Maria Helena Barcellos  
- Larissa Magalhães  
- Italo Tomazeli  

##  Instituição

**Engenharia da Computação - FAESA**

##  Licença

Uso educacional e acadêmico.
