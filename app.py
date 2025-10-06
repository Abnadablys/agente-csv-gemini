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

# Suprime logs chatos do Google/TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carrega .env
load_dotenv()

# --- Configuração da API Gemini (funciona local + online) ---
if "GOOGLE_API_KEY" in st.secrets:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ Chave API do Gemini não encontrada. Adicione em Settings → Secrets (ou .env local).")
else:
    os.environ["GOOGLE_API_KEY"] = API_KEY  # Necessário para o LangChain/Google funcionar

# Função aproximadora de tokens (1 token ≈ 4 chars)
def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1  # Aproximação conservadora

class AgenteGeminiCSV:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.df = None
        self.memory = []
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.prompt_template = PromptTemplate(
            input_variables=["data_summary", "sample_data", "history", "question"],
            template="""
            Você é um agente analista de dados. Responda a pergunta sobre o CSV, calculando tudo necessário (médias, medianas, desvios, outliers, correlações, etc.) baseado no resumo full e amostra. Use os dados exatos fornecidos.

            Resumo completo (calcule com mean= média, 50%=mediana, std=desvio padrão, min/max pra outliers): {data_summary}
            Amostra relevante (use pra verificação): {sample_data}
            Histórico recente: {history}

            Pergunta: {question}

            Instruções:
            - Calcule e responda exato (ex.: 'Média de Amount: 240.90; Mediana: 122.21').
            - Seja breve: 2-4 frases.
            - Sugira APENAS UM gráfico específico.
            - Finalize com 2-3 conclusões chave.
            """
        )
        self.chain = self.prompt_template | self.llm

    def load_csv(self, uploaded_file):
        """Carrega CSV do upload."""
        try:
            self.df = pd.read_csv(uploaded_file)
            self.memory = []  # Limpa ao novo upload
            st.success(f"✅ CSV carregado: {len(self.df)} linhas, colunas: {list(self.df.columns)}")
            return True
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")
            return False

    def get_data_summary(self) -> str:
        if self.df is None:
            return "Sem dados."
        summary = f"Linhas: {len(self.df)}\nColunas: {len(self.df.columns)}\nTipos: {dict(self.df.dtypes)}\n"
        
        # Resumo full sem truncagem (Gemini aguenta; ~1k tokens max)
        num_cols = self.df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            full_stats = self.df[num_cols].describe().round(2)
            summary += f"Stats full numéricas:\n{full_stats.to_string()}\n"
        
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            vc = self.df[col].value_counts().to_dict()  # Full value_counts
            summary += f"{col} frequências full: {vc}\n"
        
        return summary

    def get_relevant_sample(self, question: str, sample_size: int = 20) -> str:  # Maior amostra pra mais dados
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
            sample_df = self.df.sample(n=min(sample_size * 2, len(self.df)))
            num_cols = self.df.select_dtypes(include=['number']).columns
            if len(num_cols) > 0:
                sample_df = sample_df.sort_values(num_cols[0], ascending=False).head(sample_size)
        return f"Amostra ({len(sample_df)} linhas):\n{sample_df.head(sample_size).to_string(index=False)}"  # Sem truncagem

    def suggest_and_generate_graph(self, question: str, answer: str) -> Tuple[str, go.Figure]:
        """Gera gráfico variado baseado na pergunta + texto da resposta."""
        q_lower = question.lower()
        answer_lower = answer.lower()

        # Parser melhor: Extrai sugestão completa
        suggestion = ""
        graph_match = re.search(r"sugira\s+um\s+gráfico\s+de\s+['\"]([^'\"]+)['\"]", answer_lower) or \
                      re.search(r"gráfico\s+de\s+(['\"]([^'\"]+)['\"])", answer_lower)
        if graph_match:
            suggestion = graph_match.group(1) or graph_match.group(2)
        else:
            # Fallback: Usa keywords da pergunta
            if 'pizza' in q_lower or 'proporção' in q_lower:
                suggestion = "gráfico de pizza de Class"
            elif 'scatter' in q_lower or 'relação' in q_lower:
                suggestion = "scatter de V1 vs V2 por Class"
            elif 'barras' in q_lower or 'contagem' in q_lower:
                suggestion = "gráfico de barras de Class"
            elif 'box' in q_lower or 'variabilidade' in q_lower:
                suggestion = "boxplot de Amount por Class"
            elif 'linha' in q_lower or 'tendência' in q_lower:
                suggestion = "gráfico de linhas de Amount por Time"
            elif 'heatmap' in q_lower or 'correlações' in q_lower:
                suggestion = "heatmap de correlações V1-V3"
            else:
                suggestion = "histograma de Amount"

        suggest_lower = suggestion.lower()

        # Cases explícitos com variação
        num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist() or ['Class']

        if 'pizza' in suggest_lower or 'proporção' in suggest_lower:
            labels = self.df[cat_cols[0]].value_counts().index
            values = self.df[cat_cols[0]].value_counts().values
            fig = px.pie(values=values, names=labels, title=f"Proporção de {cat_cols[0]}")
        elif 'barras' in suggest_lower or 'contagem' in suggest_lower or 'frequentes' in suggest_lower:
            col = 'Class' if 'class' in suggest_lower else cat_cols[0]
            counts = self.df[col].value_counts()
            fig = px.bar(x=counts.index, y=counts.values, title=f"Contagem de {col}")
        elif 'box' in suggest_lower or 'variabilidade' in suggest_lower:
            y_col = 'Amount' if 'amount' in suggest_lower else num_cols[0]
            x_col = 'Class' if 'class' in suggest_lower else None
            if x_col:
                fig = px.box(self.df, x=x_col, y=y_col, title=f"Boxplot de {y_col} por {x_col}")
            else:
                fig = px.box(self.df, y=y_col, title=f"Boxplot de {y_col}")
        elif 'scatter' in suggest_lower or 'dispersão' in suggest_lower or 'relação' in suggest_lower or 'correlação' in suggest_lower:
            x_candidates = ['V1', 'V2', 'V3', 'Time']
            y_candidates = ['Amount', 'V10', 'V20']
            x_col = next((c for c in x_candidates if c in suggest_lower), random.choice(x_candidates))
            y_col = next((c for c in y_candidates if c in suggest_lower), random.choice(y_candidates))
            color_col = 'Class' if 'class' in suggest_lower else None
            fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, title=f"Scatter {x_col} vs {y_col}" + (f" por {color_col}" if color_col else ""))
        elif 'linha' in suggest_lower or 'tendência' in suggest_lower or 'time' in suggest_lower:
            x_col = 'Time' if 'time' in suggest_lower else num_cols[0]
            y_col = 'Amount' if 'amount' in suggest_lower else num_cols[1 % len(num_cols)]
            df_sorted = self.df.sort_values(x_col)
            fig = px.line(df_sorted, x=x_col, y=y_col, title=f"Tendência de {y_col} por {x_col}")
        elif 'heatmap' in suggest_lower or 'correlações' in suggest_lower or 'cruzada' in suggest_lower:
            if 'cruzada' in suggest_lower:
                crosstab = pd.crosstab(self.df['Class'], pd.cut(self.df['Amount'], bins=5))
                fig = px.imshow(crosstab, title="Heatmap de Tabela Cruzada (Class vs Intervalos de Amount)")
            else:
                top_cols = num_cols[:4]
                corr = self.df[top_cols].corr()
                fig = px.imshow(corr, title="Heatmap de Correlações")
        elif 'violino' in suggest_lower or 'densidade' in suggest_lower:
            y_col = 'Amount' if 'amount' in suggest_lower else num_cols[0]
            x_col = 'Class' if 'class' in suggest_lower else None
            if x_col:
                fig = px.violin(self.df, x=x_col, y=y_col, title=f"Violin de {y_col} por {x_col}")
            else:
                fig = px.violin(self.df, y=y_col, title=f"Violin de {y_col}")
        else:
            # Default variado
            random.seed(hash(question) % 100)
            graph_types = ['histograma', 'bar', 'scatter']
            type_choice = random.choice(graph_types)
            if type_choice == 'histograma':
                col = random.choice(num_cols)
                fig = px.histogram(self.df, x=col, title=f"Histograma de {col}")
            elif type_choice == 'bar':
                top_df = self.df.nlargest(10, 'Amount')[['Time', 'Amount', 'Class']]
                fig = px.bar(top_df, x='Time', y='Amount', color='Class', title="Top 10 Amounts por Class")
            else:
                cols = random.sample(num_cols, 2)
                fig = px.scatter(self.df, x=cols[0], y=cols[1], title=f"Scatter {cols[0]} vs {cols[1]}")

        return suggestion, fig

    def answer_question(self, question: str) -> str:
        if self.df is None:
            return "Nenhum CSV carregado."
        
        # Full pro Gemini: Envia tudo, calcula lá
        summary = self.get_data_summary()
        sample = self.get_relevant_sample(question)
        history = "\n".join([f"P: {h['pergunta']}\nR: {h['resposta'][:100]}..." for h in self.memory[-2:]])

        try:
            # Prompt full
            full_prompt = self.prompt_template.format(
                data_summary=summary,
                sample_data=sample,
                history=history,
                question=question
            )
            
            # Estima tokens input
            input_tokens = estimate_tokens(full_prompt)
            
            # Chama Gemini
            result = self.llm.invoke(full_prompt)
            answer = result.content.strip()  # Gemini retorna content
            
            # Estima tokens output
            output_tokens = estimate_tokens(answer)
            total_tokens = input_tokens + output_tokens
            
            # Log no Streamlit sidebar pra você ver
            with st.sidebar.expander("📊 Token Usage (Aproximado)"):
                st.metric("Input Tokens", input_tokens)
                st.metric("Output Tokens", output_tokens)
                st.metric("Total Tokens", total_tokens)
            
            self.memory.append({"pergunta": question, "resposta": answer})
            time.sleep(2)
            return answer
        except Exception as e:
            return f"Erro: {e}"

# Interface Streamlit
def main():
    st.set_page_config(page_title="Agente CSV com Gemini", layout="wide")
    st.title("🚀 Agente Analista de CSV com Gemini 2.5 Flash")
    st.markdown("Upload um CSV, faça perguntas e veja respostas + gráficos + tokens usados!")

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload CSV")
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        if uploaded_file is not None:
            agente = AgenteGeminiCSV()
            if agente.load_csv(uploaded_file):
                st.success("Carregado!")
                st.subheader("Resumo Rápido")
                st.text(agente.get_data_summary()[:400] + "...")

    # Chat
    if 'agente' not in st.session_state:
        st.session_state.agente = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Render histórico seguro
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("fig") and hasattr(message["fig"], 'to_dict'):
                st.plotly_chart(message["fig"], use_container_width=True, key=f"hist_graph_{i}")
            elif message.get("fig"):
                st.warning("Gráfico inválido – ignorando.")

    if st.session_state.agente is None and uploaded_file is not None:
        st.session_state.agente = agente

    if prompt := st.chat_input("Faça uma pergunta sobre o CSV..."):
        if st.session_state.agente is None:
            st.warning("Upload um CSV primeiro!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analisando com Gemini..."):
                    answer = st.session_state.agente.answer_question(prompt)
                    suggestion, fig = st.session_state.agente.suggest_and_generate_graph(prompt, answer)
                    
                    st.markdown(f"**{answer}**")
                    
                    if fig:
                        st.markdown(f"📊 Gráfico: {suggestion}")
                        st.plotly_chart(fig, use_container_width=True, key=f"graph_{len(st.session_state.messages)}_{int(time.time())}")
                    
                    full_response = answer + ("\n\n📊 Gráfico adicionado!" if fig else "")

                st.session_state.messages.append({"role": "assistant", "content": full_response, "fig": fig if fig else None})

if __name__ == "__main__":
    main()