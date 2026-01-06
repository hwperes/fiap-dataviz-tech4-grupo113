# ============================
# IMPORTAÇÕES
# ============================
import io
import unicodedata
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st

# ============================
# CONFIGURAÇÕES GERAIS
# ============================
validar_shap = 'n'

st.set_page_config(
    page_title="Análise de Risco de Obesidade",
    layout="wide"
)

st.title("🍟 Análise de Risco de Obesidade")
st.info("Este aplicativo visa evidenciar as situações de risco analisadas de acordo com o banco de dados!")

# ============================
# FUNÇÕES AUXILIARES
# ============================

def ordenar_opcoes(lista):
    def normalizar(texto):
        if isinstance(texto, str):
            return unicodedata.normalize('NFKD', texto)\
                .encode('ASCII', 'ignore')\
                .decode('utf-8')\
                .lower()
        return str(texto)
    return sorted(lista, key=normalizar)


def traduzir_nomes_features(feature_names_raw):
    """
    Traduz features do OneHotEncoder preservando:
    - contexto da variável
    - categoria
    - unicidade absoluta (exigência do SHAP)
    """

    mapa_base = {
        'imc': 'IMC',
        'idade': 'Idade',
        'genero': 'Gênero',
        'b_historico_familiar': 'Histórico Familiar',
        'b_fuma': 'Fumo',
        'b_come_alimentos_caloricos': 'Alimentos Calóricos',
        'b_monitora_calorias': 'Monitoramento de Calorias',
        'qtd_refeicao': 'Refeições',
        'qtd_vegetais': 'Vegetais',
        'qtd_agua': 'Água',
        'qtd_atv_fisicas': 'Atividade Física',
        'qtd_tmp_na_internet': 'Tempo em Telas',
        'freq_come_fora_refeicao': 'Comer Fora de Hora',
        'freq_alcool': 'Álcool',
        'meio_de_transporte': 'Transporte'
    }

    nomes_traduzidos = []
    contador = {}

    for nome in feature_names_raw:
        nome_limpo = (
            nome.replace('num__', '')
                .replace('cat__', '')
                .replace('bin__', '')
        )

        if '_' in nome_limpo:
            base, categoria = nome_limpo.split('_', 1)
            base_pt = mapa_base.get(base, base.replace('_', ' ').title())
            categoria_pt = categoria.replace('_', ' ').title()
            nome_final = f"{base_pt}: {categoria_pt}"
        else:
            nome_final = mapa_base.get(nome_limpo, nome_limpo.title())

        # Blindagem contra duplicidade (obrigatório para SHAP)
        if nome_final in contador:
            contador[nome_final] += 1
            nome_final = f"{nome_final} ({contador[nome_final]})"
        else:
            contador[nome_final] = 1

        nomes_traduzidos.append(nome_final)

    return nomes_traduzidos


# ============================
# LOAD MODEL
# ============================

@st.cache_resource
def load_model():
    try:
        return joblib.load("risco_obesidade_random_forest.joblib")
    except FileNotFoundError:
        url = (
            "https://github.com/hwperes/fiap-dataviz-tech4-grupo113/"
            "raw/main/Modelos/risco_obesidade_random_forest.joblib"
        )
        response = requests.get(url)
        if response.status_code == 200:
            return joblib.load(io.BytesIO(response.content))
    return None


@st.cache_resource
def _get_shap_explainer(classifier):
    return shap.TreeExplainer(classifier)

# ============================
# INPUT DO USUÁRIO (FORMULÁRIO)
# ============================

def get_user_input_features():

    st.header("1. Dados Pessoais")

    col1, col2 = st.columns(2)

    with col1:
        idade = st.number_input("Idade", 10, 100, 30)
        altura = st.number_input("Altura (m)", 1.0, 2.5, 1.70)

    with col2:
        peso = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
        genero_label = st.selectbox(
            "Gênero",
            ordenar_opcoes(["Masculino", "Feminino"])
        )

    genero = 1 if genero_label == "Feminino" else 0
    imc = int(np.ceil(peso / (altura ** 2)))

    st.info(f"IMC Calculado: **{imc} kg/m²**")
    st.markdown("---")

    st.header("2. Histórico e Hábitos")

    col3, col4 = st.columns(2)

    with col3:
        historico = st.radio("Histórico familiar?", ["Sim", "Não"])
        fuma = st.radio("Fuma?", ["Sim", "Não"])

    with col4:
        caloricos = st.radio("Consome calóricos?", ["Sim", "Não"])
        monitora = st.radio("Monitora calorias?", ["Sim", "Não"])

    st.markdown("---")

    st.header("3. Estilo de Vida")

    qtd_atv_fisicas = st.selectbox(
        "Atividade física",
        ["Sedentario", "Baixa_frequencia", "Moderada_frequencia", "Alta_frequencia"]
    )

    qtd_tmp_na_internet = st.selectbox(
        "Tempo em telas",
        ["Uso_baixo", "Uso_moderado", "Uso_intenso"]
    )

    meio_de_transporte = st.selectbox(
        "Transporte principal",
        ordenar_opcoes([
            "Public_Transportation", "Walking",
            "Automobile", "Bike", "Motorbike"
        ])
    )

    data = {
        'idade': idade,
        'genero': genero,
        'qtd_refeicao': 'Tres_refeicoes_principais_por_dia',
        'qtd_vegetais': 'Sempre',
        'qtd_agua': 'Consumo_adequado',
        'qtd_atv_fisicas': qtd_atv_fisicas,
        'qtd_tmp_na_internet': qtd_tmp_na_internet,
        'b_fuma': 1 if fuma == "Sim" else 0,
        'b_come_alimentos_caloricos': 1 if caloricos == "Sim" else 0,
        'b_monitora_calorias': 1 if monitora == "Sim" else 0,
        'b_historico_familiar': 1 if historico == "Sim" else 0,
        'freq_come_fora_refeicao': 'Sometimes',
        'freq_alcool': 'Sometimes',
        'meio_de_transporte': meio_de_transporte,
        'imc': imc
    }

    return pd.DataFrame(data, index=[0])


# ============================
# SHAP
# ============================

def gerar_explicacao_shap(model, input_df):

    preprocessor = model.named_steps['preprocess']
    classifier = model.named_steps['clf']

    X_transformed = preprocessor.transform(input_df)
    feature_names_raw = preprocessor.get_feature_names_out()
    feature_names_pt = traduzir_nomes_features(feature_names_raw)

    explainer = _get_shap_explainer(classifier)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = feature_names_pt

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        shap_values[0, :, 1],
        max_display=10,
        show=False
    )

    return plt.gcf()


# ============================
# MAIN
# ============================

def main():

    model = load_model()
    input_df = get_user_input_features()

    if st.button("🔍 Realizar Predição", use_container_width=True):

        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)

        st.markdown("---")
        st.header("Resultado")

        if prediction[0] == 1:
            st.error("⚠️ ALTO RISCO DE OBESIDADE")
        else:
            st.success("✅ BAIXO RISCO DE OBESIDADE")

        st.metric(
            "Probabilidade de Risco",
            f"{proba[0][1] * 100:.1f}%"
        )

        st.markdown("---")
        st.header("Fatores de Influência (SHAP)")

        fig = gerar_explicacao_shap(model, input_df)
        st.pyplot(fig)


if __name__ == "__main__":
    mainmain()

