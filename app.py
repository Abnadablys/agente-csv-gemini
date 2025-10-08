# ========================================================
# 🚀 AGENTE ANALISTA DE DADOS CSV COM GEMINI (Google)
# ========================================================
# Este app em Streamlit permite:
# - Fazer upload de um arquivo CSV
# - Fazer perguntas em linguagem natural sobre os dados
# - Obter respostas analíticas via IA (Gemini)
# - Gerar gráficos automaticamente (bar, scatter, box, heatmap etc.)
# ========================================================

import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Tuple
import re
import time
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import random

# Oculta logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Carrega variáveis do .env
load_dotenv()

# --------------------------------------------------------
# 🔑 CONFIGURAÇÃO DA CHAVE DA API GEMINI
# --------------------------------------------------------
try:
    # Se estiver rodando em nuvem (Streamlit Cloud), busca em st.secrets
    if "GOOGLE_API_KEY" in st.secrets:
        API_KEY = st.secrets["GOOGLE_API_KEY"]
        print("Using st.secrets (cloud)")
except Exception as e:
    # Caso contrário, tenta buscar localmente
    print(f"st.secrets not available locally: {e} - using .env")

# Busca no ambiente local se ainda não foi encontrado
if 'API_KEY' not in locals():
    API_KEY = os.getenv('GOOGLE_API_KEY')

# Se a chave não for encontrada, exibe erro e interrompe o app
if not API_KEY:
    st.error("❌ Chave API do Gemini não encontrada. Adicione GOOGLE_API_KEY no .env ou Secrets.")
    st.stop()
else:
    os.environ["GOOGLE_API_KEY"] = API_KEY


# --------------------------------------------------------
# 🔢 Função simples para estimar quantidade de tokens
# --------------------------------------------------------
def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


# --------------------------------------------------------
# 🔍 Detecta automaticamente o tipo de gráfico ideal
# --------------------------------------------------------
def choose_chart_type(df: pd.DataFrame, question: str) -> str:
    q = question.lower()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()

    # Define tipo de gráfico com base em palavras-chave da pergunta
    if 'tendência' in q or 'evolução' in q or 'tempo' in q or 'linha' in q:
        return 'line'
    elif 'proporção' in q or 'porcentagem' in q or 'pizza' in q:
        return 'pie'
    elif 'comparação' in q or 'ranking' in q or 'maior' in q or 'barras' in q:
        return 'bar'
    elif 'relação' in q or 'correlação' in q or 'dispersão' in q or 'scatter' in q:
        return 'scatter'
    elif 'variabilidade' in q or 'distribuição' in q or 'box' in q:
        return 'box'
    elif 'heatmap' in q or 'matriz' in q:
        return 'heatmap'
    else:
        # Se não conseguir identificar, faz uma escolha razoável
        if len(cat_cols) and len(num_cols):
            return 'bar'
        elif len(num_cols) > 1:
            return 'scatter'
        else:
            return 'histogram'


# --------------------------------------------------------
# 🧠 CLASSE PRINCIPAL: AGENTE GEMINI PARA CSV
# --------------------------------------------------------
class AgenteGeminiCSV:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        # Armazena o dataframe e histórico de perguntas/respostas
        self.df = None
        self.memory = []
        # Configura o modelo da Google (Gemini)
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        # Modelo de prompt para as interações com a IA
        self.prompt_template = PromptTemplate(
            input_variables=["data_summary", "sample_data", "history", "question"],
            template="""
            Você é um analista de dados. Responda à pergunta sobre o CSV com base no resumo e na amostra.
            Calcule médias, medianas, desvios, outliers, correlações e o que for necessário.

            Resumo: {data_summary}
            Amostra: {sample_data}
            Histórico recente: {history}
            Pergunta: {question}

            - Seja direto e técnico (2-4 frases).
            - Só sugira gráfico se for realmente útil.
            - Caso contrário, apenas responda.
            - Termine com 2 conclusões breves.
            """
        )
        self.chain = self.prompt_template | self.llm

    # --------------------------------------------
    # 📂 Carrega o arquivo CSV
    # --------------------------------------------
    def load_csv(self, uploaded_file):
        try:
            self.df = pd.read_csv(uploaded_file)
            self.memory = []
            st.success(f"✅ CSV carregado: {len(self.df)} linhas, colunas: {list(self.df.columns)}")
            return True
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")
            return False

    # --------------------------------------------
    # 🧾 Gera um resumo do dataset
    # --------------------------------------------
    def get_data_summary(self) -> str:
        if self.df is None:
            return "Sem dados."
        summary = f"Linhas: {len(self.df)}\nColunas: {len(self.df.columns)}\nTipos: {dict(self.df.dtypes)}\n"
        num_cols = self.df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            full_stats = self.df[num_cols].describe().round(2)
            summary += f"Stats numéricas:\n{full_stats.to_string()}\n"
        return summary

    # --------------------------------------------
    # 🔎 Extrai uma amostra relevante do CSV
    # --------------------------------------------
    def get_relevant_sample(self, question: str, sample_size: int = 15) -> str:
        if self.df is None:
            return "Sem dados."
        # Busca palavras-chave da pergunta
        keywords = re.findall(r'\w+', question.lower())
        sample_df = self.df.copy()
        text_cols = self.df.select_dtypes(include=['object']).columns
        # Filtra por colunas de texto que contenham palavras da pergunta
        for col in text_cols:
            mask = self.df[col].astype(str).str.contains('|'.join(keywords), case=False, na=False)
            if mask.any():
                sample_df = sample_df[mask]
        # Se não encontrar nada relevante, usa amostra aleatória
        if len(sample_df) == 0 or len(sample_df) > sample_size * 2:
            sample_df = self.df.sample(n=min(sample_size, len(self.df)))
        return f"Amostra ({len(sample_df)} linhas):\n{sample_df.head(sample_size).to_string(index=False)}"

    # --------------------------------------------
    # 💬 Envia a pergunta ao modelo e obtém resposta
    # --------------------------------------------
    def answer_question(self, question: str) -> str:
        if self.df is None:
            return "Nenhum CSV carregado."

        summary = self.get_data_summary()
        sample = self.get_relevant_sample(question)
        # Usa histórico das últimas 2 perguntas
        history = "\n".join([f"P: {h['pergunta']}\nR: {h['resposta'][:100]}..." for h in self.memory[-2:]])

        try:
            full_prompt = self.prompt_template.format(
                data_summary=summary, sample_data=sample, history=history, question=question
            )
            input_tokens = estimate_tokens(full_prompt)
            # Envia prompt ao Gemini
            result = self.llm.invoke(full_prompt)
            answer = result.content.strip()
            output_tokens = estimate_tokens(answer)
            total_tokens = input_tokens + output_tokens

            # Mostra estimativa de tokens na barra lateral
            with st.sidebar.expander("📊 Tokens (estimado)"):
                st.metric("Input Tokens", input_tokens)
                st.metric("Output Tokens", output_tokens)
                st.metric("Total Tokens", total_tokens)

            # Armazena pergunta e resposta
            self.memory.append({"pergunta": question, "resposta": answer})
            time.sleep(1)
            return answer
        except Exception as e:
            return f"Erro: {e}"

    # --------------------------------------------
    # 📈 Gera gráficos automaticamente
    # --------------------------------------------
    def generate_graph(self, question: str, chart_type: str) -> go.Figure:
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude='number').columns.tolist() or ['Class']

        if not num_cols:
            st.warning("Nenhuma coluna numérica para gráfico.")
            return None

        # 🍕 Gráfico de pizza
        if chart_type == 'pie':
            labels = self.df[cat_cols[0]].value_counts().index
            values = self.df[cat_cols[0]].value_counts().values
            fig = px.pie(values=values, names=labels, title=f"Proporção de {cat_cols[0]}")

        # 📊 Gráfico de barras
        elif chart_type == 'bar':
            col = cat_cols[0]
            counts = self.df[col].value_counts().head(10)
            fig = px.bar(x=counts.index, y=counts.values, title=f"Top 10 de {col}")

        # 🔵 Gráfico de dispersão
        elif chart_type == 'scatter':
            if len(num_cols) < 2:
                return None
            fig = px.scatter(self.df.sample(min(1000, len(self.df))), x=num_cols[0], y=num_cols[1],
                             title=f"Dispersão {num_cols[0]} vs {num_cols[1]}")

        # 📦 Boxplot
        elif chart_type == 'box':
            y_col = num_cols[0]
            x_col = cat_cols[0] if cat_cols else None
            fig = px.box(self.df, x=x_col, y=y_col, title=f"Boxplot de {y_col} por {x_col}" if x_col else f"Boxplot de {y_col}")

        # 📈 Gráfico de linha (tendência)
        elif chart_type == 'line':
            if 'Time' in self.df.columns:
                df_sorted = self.df.sort_values('Time').head(5000)
                fig = px.line(df_sorted, x='Time', y=num_cols[0], title=f"Tendência de {num_cols[0]} por Time")
            else:
                fig = px.line(self.df.head(5000), y=num_cols[0], title=f"Tendência de {num_cols[0]}")

        # 🌡️ Heatmap de correlação
        elif chart_type == 'heatmap':
            top_cols = num_cols[:4]
            corr = self.df[top_cols].corr()
            fig = px.imshow(corr, title="Heatmap de Correlações")

        # 📉 Histograma (distribuição)
        else:
            fig = px.histogram(self.df, x=num_cols[0], title=f"Distribuição de {num_cols[0]}")

        return fig


# --------------------------------------------------------
# 🖥️ INTERFACE PRINCIPAL DO STREAMLIT
# --------------------------------------------------------
def main():
    st.set_page_config(page_title="Agente CSV com Gemini", layout="wide")
    st.title("🚀 Agente Analista de Dados CSV")

    # --- Barra lateral para upload ---
    with st.sidebar:
        st.header("📁 Upload CSV")
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        if uploaded_file is not None:
            agente = AgenteGeminiCSV()
            if agente.load_csv(uploaded_file):
                st.success("Carregado!")
                st.subheader("Resumo Rápido")
                st.text(agente.get_data_summary()[:400] + "...")

    # Inicializa variáveis de sessão
    if 'agente' not in st.session_state:
        st.session_state.agente = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Renderiza o histórico do chat
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("fig"):
                st.plotly_chart(message["fig"], use_container_width=True, key=f"graph_{i}")

    # Vincula o agente carregado à sessão
    if st.session_state.agente is None and uploaded_file is not None:
        st.session_state.agente = agente

    # --- Caixa de chat (entrada do usuário) ---
    if prompt := st.chat_input("Faça uma pergunta sobre o CSV..."):
        if st.session_state.agente is None:
            st.warning("Faça o upload de um CSV primeiro.")
        else:
            # Exibe pergunta no chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Exibe resposta da IA
            with st.chat_message("assistant"):
                with st.spinner("Analisando com Gemini..."):
                    answer = st.session_state.agente.answer_question(prompt)
                    st.markdown(f"**{answer}**")

                    # Verifica se deve gerar gráfico (pelo prompt ou resposta)
                    should_plot = bool(re.search(r"gráfic|visualiza|plot|diagrama", answer.lower()) or
                                       re.search(r"gráfic|visualiza|plot|diagrama", prompt.lower()))

                    if should_plot:
                        chart_type = choose_chart_type(st.session_state.agente.df, prompt)
                        fig = st.session_state.agente.generate_graph(prompt, chart_type)
                        if fig:
                            st.markdown(f"📊 Gráfico gerado automaticamente: **{chart_type}**")
                            st.plotly_chart(fig, use_container_width=True)
                            full_response = answer + f"\n\n📊 Gráfico ({chart_type}) adicionado!"
                        else:
                            full_response = answer
                    else:
                        full_response = answer

                # Armazena resposta e gráfico no histórico
                st.session_state.messages.append({"role": "assistant", "content": full_response,
                                                  "fig": fig if should_plot else None})


# Executa o app
if __name__ == "__main__":
    main()
