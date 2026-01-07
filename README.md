# 🧠 Modelo de Machine Learning para Previsão de Obesidade

![Status](https://img.shields.io/badge/status-concluído-success)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)

---

## 📋 Sobre o Projeto

Este repositório reúne todo o desenvolvimento do **Tech Challenge – Fase 4 (Data Visualization & Production Models)** da **Pós-Graduação em Data Analytics (FIAP + Alura)**.

O desafio consiste em desenvolver um **modelo preditivo de obesidade**, utilizando **Machine Learning**, além de:

- Uma **aplicação interativa em Streamlit**
- Um **painel analítico no Power BI**
- Uma base de dados tratada, documentada e reproduzível

A solução foi construída seguindo **boas práticas de Engenharia de Machine Learning**, com pipeline completo, versionamento de modelos e foco em **explicabilidade para apoio à decisão clínica**.

---

## 🎯 Objetivos do Projeto

- Realizar análise exploratória da base `Obesity.csv`
- Executar tratamento, limpeza e tradução dos dados
- Construir pipeline de pré-processamento e feature engineering
- Treinar e avaliar modelos preditivos (acurácia ≥ 75%)
- Selecionar o melhor modelo (Random Forest)
- Implementar aplicação preditiva em Streamlit
- Criar base analítica para consumo no Power BI
- Publicar a solução com documentação e reprodutibilidade

---

## 🏥 Contexto do Problema

A obesidade é uma condição multifatorial relacionada a hábitos alimentares, estilo de vida, fatores genéticos e ambientais.

Antecipar o **risco de obesidade** auxilia profissionais da saúde em:
- Ações preventivas
- Apoio ao diagnóstico
- Tomada de decisão clínica mais assertiva

O modelo utiliza variáveis relacionadas a:
- Alimentação
- Atividade física
- Consumo de água e álcool
- Tabagismo
- Uso de dispositivos eletrônicos
- Dados antropométricos (idade, peso, altura)

---

## 🤖 Modelo de Machine Learning

- **Algoritmo:** Random Forest Classifier
- **Pipeline completo com:**
  - Padronização de variáveis numéricas
  - One-Hot Encoding de variáveis categóricas
  - Classificação supervisionada
- **Métricas avaliadas:**
  - Acurácia
  - F1-Score
  - Matriz de Confusão
- **Explicabilidade:** SHAP (Waterfall Plot individual)

📦 Modelo versionado em:
- `model_obesity.joblib`
- `model_obesity.pkl`

---

## 🚀 Aplicação Streamlit

A aplicação preditiva está disponível em:

👉 **https://fiap-dataviz-tech4-grupo113.streamlit.app/**

Funcionalidades:
- Questionário interativo
- Cálculo automático de IMC
- Predição do risco de obesidade
- Probabilidade associada
- Visualização dos fatores que mais influenciaram a decisão do modelo (SHAP)

---

## 📊 Dashboard Analítico (Power BI)

Foi construída uma base analítica em português, pronta para consumo no Power BI, contendo:

- IMC
- Faixa etária
- Nível de obesidade real e previsto
- Probabilidade do modelo
- Indicadores de desempenho

Arquivo principal:
- `obesidade_analiticoBI_ptbr.csv`

---

## 🗂 Estrutura do Projeto

```bash
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/
│   │   └── Obesity.csv
│   └── processed/
│       └── obesidade_analiticoBI_ptbr.csv
├── models/
│   ├── model_obesity.joblib
│   └── model_obesity.pkl
├── notebooks/
│   └── tech_challenge_codigo.ipynb
├── references/
│   └── dicionario_obesity_fiap.pdf
├── app.py
├── requirements.txt
├── LICENSE
└── README.md
