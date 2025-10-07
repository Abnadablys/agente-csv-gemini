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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
load_dotenv()

# --- Configuração da API ---
try:
    if "GOOGLE_API_KEY" in st.secrets:
        API_KEY = st.secrets["GOOGLE_API_KEY"]
        print("Using st.secrets (cloud)")
except Exception as e:
    print(f"st.secrets not available locally: {e} - using .env")

if 'API_KEY' not in locals():
    API_KEY = os.getenv('GOOGLE_API_KEY')

if not API_KEY:
    st.error("❌ Chave API do Gemini não encontrada. Adicione GOOGLE_API_KEY no .env ou Secrets.")
    st.stop()
else:
    os.environ["GOOGLE_API_KEY"] = API_KEY


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


# --------------------------------------------------------
# 🔍 Função leve para decidir o tipo de gráfico
# --------------------------------------------------------
def choose_chart_type(df: pd.DataFrame, question: str) -> str:
    q = question.lower()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()

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
        if len(cat_cols) and len(num_cols):
            return 'bar'
        elif len(num_cols) > 1:
            return 'scatter'
        else:
            return 'histogram'


# --------------------------------------------------------
# 🧠 Classe principal do agente Gemini
# --------------------------------------------------------
class AgenteGeminiCSV:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.df = None
        self.memory = []
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
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
            - Só sugira gráfico se for realmente útil (ex.: "Um gráfico de barras ajudaria a visualizar isso.").
            - Caso contrário, apenas responda.
            - Termine com 2 conclusões breves.
            """
        )
        self.chain = self.prompt_template | self.llm

    def load_csv(self, uploaded_file):
        try:
            self.df = pd.read_csv(uploaded_file)
            self.memory = []
            st.success(f"✅ CSV carregado: {len(self.df)} linhas, colunas: {list(self.df.columns)}")
            return True
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")
            return False

    def get_data_summary(self) -> str:
        if self.df is None:
            return "Sem dados."
        summary = f"Linhas: {len(self.df)}\nColunas: {len(self.df.columns)}\nTipos: {dict(self.df.dtypes)}\n"
        num_cols = self.df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            full_stats = self.df[num_cols].describe().round(2)
            summary += f"Stats numéricas:\n{full_stats.to_string()}\n"
        return summary

    def get_relevant_sample(self, question: str, sample_size: int = 15) -> str:
        if self.df is None:
            return "Sem dados."
        keywords = re.findall(r'\w+', question.lower())
        sample_df = self.df.copy()
        text_cols = self.df.select_dtypes(include=['object']).columns
        for col in text_cols:
            mask = self.df[col].astype(str).str.contains('|'.join(keywords), case=False, na=False)
            if mask.any():
                sample_df = sample_df[mask]
        if len(sample_df) == 0 or len(sample_df) > sample_size * 2:
            sample_df = self.df.sample(n=min(sample_size, len(self.df)))
        return f"Amostra ({len(sample_df)} linhas):\n{sample_df.head(sample_size).to_string(index=False)}"

    def answer_question(self, question: str) -> str:
        if self.df is None:
            return "Nenhum CSV carregado."
        summary = self.get_data_summary()
        sample = self.get_relevant_sample(question)
        history = "\n".join([f"P: {h['pergunta']}\nR: {h['resposta'][:100]}..." for h in self.memory[-2:]])

        try:
            full_prompt = self.prompt_template.format(
                data_summary=summary, sample_data=sample, history=history, question=question
            )
            input_tokens = estimate_tokens(full_prompt)
            result = self.llm.invoke(full_prompt)
            answer = result.content.strip()
            output_tokens = estimate_tokens(answer)
            total_tokens = input_tokens + output_tokens

            with st.sidebar.expander("📊 Tokens (estimado)"):
                st.metric("Input Tokens", input_tokens)
                st.metric("Output Tokens", output_tokens)
                st.metric("Total Tokens", total_tokens)

            self.memory.append({"pergunta": question, "resposta": answer})
            time.sleep(1)
            return answer
        except Exception as e:
            return f"Erro: {e}"

    def generate_graph(self, question: str, chart_type: str) -> go.Figure:
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude='number').columns.tolist() or ['Class']

        if not num_cols:
            st.warning("Nenhuma coluna numérica para gráfico.")
            return None

        if chart_type == 'pie':
            labels = self.df[cat_cols[0]].value_counts().index
            values = self.df[cat_cols[0]].value_counts().values
            fig = px.pie(values=values, names=labels, title=f"Proporção de {cat_cols[0]}")

        elif chart_type == 'bar':
            col = cat_cols[0]
            counts = self.df[col].value_counts().head(10)
            fig = px.bar(x=counts.index, y=counts.values, title=f"Top 10 de {col}")

        elif chart_type == 'scatter':
            if len(num_cols) < 2:
                return None
            fig = px.scatter(self.df.sample(min(1000, len(self.df))), x=num_cols[0], y=num_cols[1],
                             title=f"Dispersão {num_cols[0]} vs {num_cols[1]}")

        elif chart_type == 'box':
            y_col = num_cols[0]
            x_col = cat_cols[0] if cat_cols else None
            fig = px.box(self.df, x=x_col, y=y_col, title=f"Boxplot de {y_col} por {x_col}" if x_col else f"Boxplot de {y_col}")

        elif chart_type == 'line':
            if 'Time' in self.df.columns:
                df_sorted = self.df.sort_values('Time').head(5000)
                fig = px.line(df_sorted, x='Time', y=num_cols[0], title=f"Tendência de {num_cols[0]} por Time")
            else:
                fig = px.line(self.df.head(5000), y=num_cols[0], title=f"Tendência de {num_cols[0]}")

        elif chart_type == 'heatmap':
            top_cols = num_cols[:4]
            corr = self.df[top_cols].corr()
            fig = px.imshow(corr, title="Heatmap de Correlações")

        else:
            fig = px.histogram(self.df, x=num_cols[0], title=f"Distribuição de {num_cols[0]}")

        return fig


# --------------------------------------------------------
# 🎛️ Interface Streamlit
# --------------------------------------------------------
def main():
    st.set_page_config(page_title="Agente CSV com Gemini", layout="wide")
    st.title("🚀 Agente Analista de Dados CSV")

    with st.sidebar:
        st.header("📁 Upload CSV")
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        if uploaded_file is not None:
            agente = AgenteGeminiCSV()
            if agente.load_csv(uploaded_file):
                st.success("Carregado!")
                st.subheader("Resumo Rápido")
                st.text(agente.get_data_summary()[:400] + "...")

    if 'agente' not in st.session_state:
        st.session_state.agente = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("fig"):
                st.plotly_chart(message["fig"], use_container_width=True, key=f"graph_{i}")

    if st.session_state.agente is None and uploaded_file is not None:
        st.session_state.agente = agente

    if prompt := st.chat_input("Faça uma pergunta sobre o CSV..."):
        if st.session_state.agente is None:
            st.warning("Faça o upload de um CSV primeiro.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analisando com Gemini..."):
                    answer = st.session_state.agente.answer_question(prompt)
                    st.markdown(f"**{answer}**")

                    # Decide automaticamente se deve gerar gráfico
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

                st.session_state.messages.append({"role": "assistant", "content": full_response,
                                                  "fig": fig if should_plot else None})


if __name__ == "__main__":
    main()
