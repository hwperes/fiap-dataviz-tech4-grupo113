# 🩺 Predição de Risco de Obesidade com Machine Learning

Este projeto utiliza **Machine Learning**, **Data Visualization** e **Deploy em Streamlit** para prever o **risco de obesidade** com base em características físicas, hábitos alimentares e estilo de vida do indivíduo.

O trabalho foi desenvolvido como parte do **Tech Challenge – Fase 4 (Data Viz & Production Models)** da **Pós-Graduação em Data Analytics (FIAP + Alura)**.

---

## 🎯 Objetivo do Projeto

Criar uma solução **end-to-end**, contemplando:

- 📊 Análise e tratamento de dados  
- 🤖 Treinamento de modelo preditivo  
- 🔍 Explicabilidade com SHAP  
- 🌐 Aplicação interativa com Streamlit  
- 📈 Base preparada para consumo no Power BI  
- 🚀 Versionamento e deploy via GitHub  

---

## 🧠 Modelo de Machine Learning

- **Algoritmo:** Random Forest Classifier  
- **Pipeline completo:**  
  - Pré-processamento (numéricos, binários e categóricos)  
  - One-Hot Encoding  
  - Classificação  
- **Saída do modelo:**  
  - Predição binária (Risco de Obesidade: Sim/Não)  
  - Probabilidade associada  
- **Explicabilidade:**  
  - SHAP (Waterfall Plot individual)  

📦 Modelos serializados:
- `.joblib` → modelo principal  
- `.pkl` → objetos auxiliares  

---

## 🖥️ Aplicação Streamlit

A aplicação permite:

- Preenchimento de um **questionário interativo**
- Cálculo automático de **IMC**
- Exibição do **resultado da predição**
- Visualização dos **fatores que mais impactaram a decisão do modelo**
- Interface amigável, organizada por seções

### Tecnologias Utilizadas
- `streamlit`
- `scikit-learn`
- `pandas`
- `numpy`
- `shap`
- `matplotlib`

---

## 📊 Power BI

Os dados foram:

- Tratados
- Traduzidos para **Português**
- Padronizados
- Exportáveis para consumo no **Power BI**

Permitindo:
- Dashboards analíticos
- Indicadores de saúde
- Análises exploratórias e executivas

---

## 📂 Estrutura do Repositório

```bash
📁 projeto-risco-obesidade
│
├── 📓 notebook/
│   └── treinamento_modelo.ipynb
│
├── 📊 data/
│   └── obesity_tratado_powerbi.csv
│
├── 🤖 model/
│   ├── modelo_risco_obesidade_random_forest.joblib
│   └── objetos_auxiliares.pkl
│
├── 🌐 app/
│   └── app_streamlit.py
│
├── 📄 requirements.txt
├── README.md
└── .gitignore
