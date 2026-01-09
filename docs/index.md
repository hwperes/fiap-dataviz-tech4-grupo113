# Predição de Risco de Obesidade

Bem-vindo à documentação oficial do projeto **Tech Challenge — Fase 4**, desenvolvido no contexto da **Pós-Graduação em Data Analytics da FIAP**.  
Este projeto aplica técnicas de Ciência de Dados e Machine Learning para apoiar profissionais da saúde na identificação do risco de obesidade.

---

## Objetivo do Projeto

O propósito central deste trabalho é construir um **modelo preditivo de obesidade** capaz de classificar indivíduos com base em características físicas, hábitos alimentares, estilo de vida e fatores comportamentais.

Além da etapa de modelagem, o projeto contempla:
- tratamento e preparação dos dados,
- avaliação comparativa de algoritmos de Machine Learning,
- disponibilização do modelo por meio de uma aplicação interativa,
- geração de insumos analíticos para apoio à tomada de decisão clínica.

---

## Abordagem da Solução

A solução foi estruturada em etapas bem definidas, seguindo boas práticas de engenharia de Machine Learning:

### Análise Exploratória dos Dados  
Realizou-se uma avaliação detalhada da base de dados de obesidade, buscando compreender a distribuição das variáveis, padrões comportamentais e possíveis relações com os níveis de obesidade.

### Preparação e Engenharia de Dados  
Foram aplicadas técnicas de:
- limpeza e padronização dos dados,
- codificação de variáveis categóricas,
- normalização das variáveis numéricas,
- criação de atributos derivados, como o Índice de Massa Corporal (IMC).

### Modelagem Preditiva  
Dois algoritmos principais foram avaliados:
- **Regressão Logística**, utilizada como modelo de referência,
- **Random Forest**, selecionado como modelo final devido ao melhor desempenho.

As métricas analisadas incluíram acurácia, precisão, recall, F1-Score e matriz de confusão.

### Aplicação Interativa  
O modelo final foi integrado a uma aplicação desenvolvida em **Streamlit**, permitindo a realização de predições de forma simples e intuitiva por usuários finais.

---

## Principais Resultados

- **Modelo Selecionado:** Random Forest Classifier  
- **Desempenho:**  
  - Acurácia superior à Regressão Logística  
  - Melhor equilíbrio entre precisão e recall  
- **Principais Variáveis Influentes:**  
  - Índice de Massa Corporal (IMC)  
  - Histórico familiar de obesidade  
  - Frequência de refeições  
  - Nível de atividade física  

Esses resultados reforçam a capacidade do modelo em identificar corretamente indivíduos em risco, minimizando falsos negativos.

---

## Equipe de Desenvolvimento

Este projeto foi desenvolvido pelo grupo:

- Fabiana Cardoso da Silva  
- Henrique do Couto Santos  
- Henrique Waideman Peres  

---

Para mais detalhes técnicos, consulte o documento de **modelagem**, o **notebook do projeto** e o **repositório oficial** com todo o código-fonte.
