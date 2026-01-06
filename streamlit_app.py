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

validar_shap = 'n'

st.set_page_config(page_title="Análise de Risco de Obesidade", layout="wide")
st.title('🍟 Análise de Risco de Obesidade')
st.info('Este aplicativo visa evidenciar as situações de risco analisadas de acordo com o banco de dados!')

# ============================
# FUNÇÕES AUXILIARES
# ============================

def ordenar_opcoes(lista):
    def normalizar(texto):
        if isinstance(texto, str):
            return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
        return str(texto)
    return sorted(lista, key=normalizar)


def traduzir_nomes_features(feature_names_raw):
    """
    Traduz nomes técnicos do OneHotEncoder garantindo unicidade
    e preservando o contexto da variável + categoria.
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
        nome_limpo = nome.replace('num__', '').replace('cat__', '').replace('bin__', '')

        if '_' in nome_limpo:
            base, categoria = nome_limpo.split('_', 1)
            base_pt = mapa_base.get(base, base.replace('_', ' ').title())
            categoria_pt = categoria.replace('_', ' ').title()
            nome_final = f"{base_pt}: {categoria_pt}"
        else:
            nome_final = mapa_base.get(nome_limpo, nome_limpo.title())

        # Blindagem contra duplicidade
        if nome_final in contador:
            contador[nome_final] += 1
            nome_final = f"{nome_final} ({contador[nome_final]})"
        else:
            contador[nome_final] = 1

        nomes_traduzidos.append(nome_final)

    return nomes_traduzidos


# ============================
# MODEL LOAD
# ============================

@st.cache_resource
def load_model():
    try:
        return joblib.load('risco_obesidade_random_forest.joblib')
    except FileNotFoundError:
        url = "https://github.com/hwperes/fiap-dataviz-tech4-grupo113/raw/main/Modelos/risco_obesidade_random_forest.joblib"
        response = requests.get(url)
        if response.status_code == 200:
            return joblib.load(io.BytesIO(response.content))
    return None


@st.cache_resource
def _get_shap_explainer(_classifier):
    return shap.TreeExplainer(_classifier)


# ============================
# SHAP
# ============================

def gerar_explicacao_shap(model, input_df):
    preprocessor = model.named_steps['preprocess']
    classifier = model.named_steps['clf']

    input_transformed = preprocessor.transform(input_df)
    feature_names_raw = preprocessor.get_feature_names_out()
    feature_names_pt = traduzir_nomes_features(feature_names_raw)

    df_mapeamento = pd.DataFrame({
        'Nome Técnico': feature_names_raw,
        'Nome Traduzido': feature_names_pt,
        'Valor': input_transformed[0]
    })

    explainer = _get_shap_explainer(classifier)
    shap_values = explainer(input_transformed)
    shap_values.feature_names = feature_names_pt

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0, :, 1], show=False, max_display=10)

    return plt.gcf(), df_mapeamento


# ============================
# MAIN
# ============================

def main():
    model = load_model()
    input_df = get_user_input_features()

    if st.button("🔍 Realizar Predição", type="primary", use_container_width=True):
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        st.markdown("---")
        st.header("Resultado da Análise")

        if prediction[0] == 1:
            st.error("⚠️ ALTO RISCO DE OBESIDADE")
        else:
            st.success("✅ BAIXO RISCO")

        st.metric("Probabilidade", f"{probability[0][1]*100:.1f}%")

        st.markdown("---")
        st.header("Fatores de Influência")

        fig, df_map = gerar_explicacao_shap(model, input_df)
        st.pyplot(fig)

        if validar_shap.lower() == 's':
            st.dataframe(df_map)


if __name__ == "__main__":
    main()
