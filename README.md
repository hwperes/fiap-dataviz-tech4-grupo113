🩺 Predição de Risco de Obesidade com Machine Learning

Este projeto utiliza Machine Learning, Data Visualization e Deploy em Streamlit para prever o risco de obesidade com base em características físicas, hábitos alimentares e estilo de vida do indivíduo.

O trabalho foi desenvolvido como parte do Tech Challenge – Fase 4 (Data Viz & Production Models) da Pós-Graduação em Data Analytics (FIAP + Alura).

🎯 Objetivo do Projeto

Criar uma solução end-to-end, contemplando:

📊 Análise e tratamento de dados

🤖 Treinamento de modelo preditivo

🔍 Explicabilidade com SHAP

🌐 Aplicação interativa com Streamlit

📈 Base preparada para consumo no Power BI

🚀 Versionamento e deploy via GitHub

🧠 Modelo de Machine Learning

Algoritmo: Random Forest Classifier

Pipeline completo:

Pré-processamento (numéricos, binários e categóricos)

One-Hot Encoding

Classificação

Saída do modelo:

Predição binária (Risco de Obesidade: Sim/Não)

Probabilidade associada

Explicabilidade:

SHAP (Waterfall Plot individual)

📦 Modelos serializados:

.joblib → modelo principal

.pkl → objetos auxiliares (se aplicável)

🖥️ Aplicação Streamlit

A aplicação permite:

Preenchimento de um questionário interativo

Cálculo automático de IMC

Exibição do resultado da predição

Visualização dos fatores que mais impactaram a decisão do modelo

Interface amigável, organizada por seções

Principais Tecnologias

streamlit

scikit-learn

pandas

numpy

shap

matplotlib

📊 Power BI

Os dados foram:

Tratados

Traduzidos para Português

Padronizados

Exportáveis para consumo no Power BI

Isso permite:

Dashboards analíticos

Acompanhamento de indicadores

Análises exploratórias e executivas

📂 Estrutura do Repositório
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

▶️ Como Executar o Projeto Localmente
1️⃣ Clonar o repositório
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

2️⃣ Criar ambiente virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3️⃣ Instalar dependências
pip install -r requirements.txt

4️⃣ Executar o Streamlit
streamlit run app/app_streamlit.py

🧪 Dados Utilizados

Dataset relacionado a hábitos alimentares, estilo de vida e saúde

Variáveis numéricas, categóricas e binárias

Dados tratados e preparados para:

Modelagem

Visualização

Consumo em BI

📌 Explicabilidade (SHAP)

O projeto utiliza SHAP para:

Explicar decisões individuais do modelo

Mostrar quais fatores:

Aumentam o risco (vermelho)

Reduzem o risco (azul)

Tornar o modelo interpretável e confiável

👨‍🎓 Contexto Acadêmico

🎓 Curso: Pós-Graduação em Data Analytics

🏫 Instituição: FIAP + Alura

📦 Entrega: Tech Challenge – Fase 4

📚 Tema: Data Visualization & Production Models

🚀 Próximos Passos (Evoluções Futuras)

Deploy em cloud (Streamlit Community / Azure / AWS)

Monitoramento de drift de dados

Registro de predições

Autenticação de usuários

Integração direta com Power BI Service

👤 Autor

Henrique Waideman Peres
📊 Data Analytics | Machine Learning | BI
🎓 FIAP
