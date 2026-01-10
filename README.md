# Modelo de Machine Learning para Previsão de Obesidade

![Status](https://img.shields.io/badge/status-concluído-success)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)

---

## Sobre o Projeto

Este repositório reúne todo o desenvolvimento do **Tech Challenge – Fase 4 (Data Visualization & Production Models)** da **Pós-Graduação em Data Analytics (FIAP + Alura)**.

O desafio consiste em desenvolver um **modelo preditivo de obesidade**, utilizando **Machine Learning**, além de:

- Uma **aplicação interativa em Streamlit**
- Um **painel analítico no Power BI**
- Uma base de dados tratada, documentada e reproduzível

A solução foi construída seguindo **boas práticas de Engenharia de Machine Learning**, com pipeline completo, versionamento de modelos e foco em **explicabilidade para apoio à decisão clínica**.

---

## Objetivos do Projeto

- Realizar análise exploratória da base `Obesity.csv`
- Executar tratamento, limpeza e tradução dos dados
- Construir pipeline de pré-processamento e feature engineering
- Treinar e avaliar modelos preditivos (acurácia ≥ 75%)
- Selecionar o melhor modelo (Random Forest)
- Implementar aplicação preditiva em Streamlit
- Criar base analítica para consumo no Power BI
- Publicar a solução com documentação e reprodutibilidade

---

## Contexto do Problema

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

## Modelo de Machine Learning

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

 Modelo versionado em:
- `model_obesity.joblib`
- `model_obesity.pkl`

---

## Aplicação Streamlit

A aplicação preditiva está disponível em:

 **https://fiap-dataviz-tech4-grupo113.streamlit.app/**

Funcionalidades:
- Questionário interativo
- Cálculo automático de IMC
- Predição do risco de obesidade
- Probabilidade associada
- Visualização dos fatores que mais influenciaram a decisão do modelo (SHAP)

---

## Dashboard Analítico (Power BI)

Foi construída uma base analítica em português, pronta para consumo no Power BI, contendo:

- IMC
- Faixa etária
- Nível de obesidade real e previsto
- Probabilidade do modelo
- Indicadores de desempenho

Arquivo principal:
- `obesidade_analiticoBI_ptbr.csv`

[Dashboard - Estudo de Obesidade - Grupo 113](https://app.powerbi.com/view?r=eyJrIjoiMWYxZWRlNmQtM2RkMC00MzQ4LWE3ZGYtNTZlNzkwMzVlMmQ0IiwidCI6ImNmNzJlMmJkLTdhMmItNDc4My1iZGViLTM5ZDU3YjA3Zjc2ZiIsImMiOjR9)

---

## Estrutura do Projeto

```bash
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/
│   │   └── Obesity.csv
│   └── processed/
│       └── obesidade_analiticoBI_ptbr.csv
├── docs/
│   ├── doc_modelagem.md
│   └── index.md
│       
├── models/
│   ├── risco_obesidade_random_forest.joblib
│   └── model_obesity.pkl
├── notebooks/
│   └── tech_challenge_codigo_fase4_grupo113.ipynb
├── references/
│   └── dicionario_obesity_fiap.pdf
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## Dados

O dicionário de dados utilizado está disponível na pasta `references/`.  
As variáveis contemplam aspectos relacionados a:

- Hábitos alimentares  
- Atividade física  
- Consumo de água e álcool  
- Tabagismo  
- Uso de dispositivos eletrônicos  
- Dados antropométricos (peso, altura e idade)  

A **variável-alvo** do modelo é **Obesity**, com níveis que variam de:

- Insufficient Weight  
- Normal Weight  
- Overweight  
- Obesity Type I  
- Obesity Type II  
- Obesity Type III

---

## Metodologia

### [1] Pré-processamento
- Tratamento e limpeza dos dados  
- Codificação de variáveis categóricas  
- Normalização e padronização  
- Feature Engineering  

### [2] Modelagem
Foram testados diferentes algoritmos de Machine Learning, incluindo:

- Random Forest (**modelo final escolhido**)  
- Logistic Regression  

**Métricas avaliadas:**
- Acurácia  
- F1-Score  
- Matriz de Confusão  

### [3] Deploy
- Aplicação preditiva desenvolvida em **Streamlit**  
- Modelo versionado em formato `.joblib`  
- Ambiente reproduzível utilizando `requirements.txt`  

---

## Dashboard Analítico

O painel analítico apresenta os principais insights extraídos dos dados, incluindo:

- IMC médio  
- Média de idade  
- Risco de obesidade  
- Nível de obesidade  

A base foi preparada e traduzida para consumo no Power BI, possibilitando análises visuais e apoio à tomada de decisão.

---

## Equipe

- **Fabiana Cardoso da Silva**  
- **Henrique do Couto Santos**  
- **Henrique Waideman Peres**

