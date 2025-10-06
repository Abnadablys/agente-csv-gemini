# 🚀 Agente Analista de CSV com Gemini 2.5 Flash

Este projeto é um **agente autônomo de análise de dados** que utiliza **Gemini 2.5 Flash**, **LangChain** e **Streamlit**.  
Ele permite o **upload de arquivos CSV**, a realização de **perguntas em linguagem natural** e a **geração automática de respostas e gráficos interativos**.

---

## 🧩 Funcionalidades Principais

✅ Upload de arquivos `.csv` até ~150MB  
✅ Análise automática dos dados (estatísticas, tipos, outliers, etc.)  
✅ Perguntas em linguagem natural com respostas detalhadas  
✅ Geração de gráficos (histogramas, boxplots, heatmaps, etc.)  
✅ Integração com **Gemini 2.5 Flash (Google AI)** via **LangChain**  
✅ Interface 100% interativa com **Streamlit**

---

## ⚙️ Estrutura do Projeto

```
📂 agente-csv-gemini/
├── app.py                  # Código principal da aplicação
├── requirements.txt        # Dependências do projeto
├── .gitignore              # Arquivos ignorados no Git
└── README.md               # Este arquivo
```

---

## 🧠 Tecnologias Utilizadas

- **Python 3.10+**
- **Streamlit**
- **LangChain**
- **Google Gemini API**
- **Plotly**
- **dotenv (para variáveis de ambiente)**

---

## 🔑 Configuração da API Gemini

Crie um arquivo chamado `.env` na raiz do projeto com o seguinte conteúdo:

```env
GOOGLE_API_KEY=coloque_sua_chave_aqui
```

> 📝 **Importante:**  
> No Streamlit Cloud, use **“Settings → Secrets”** e adicione a variável `GOOGLE_API_KEY` lá.

---

## ▶️ Como Executar Localmente

```bash
# 1. Clonar o repositório
git clone https://github.com/SEU_USUARIO/agente-csv-gemini.git
cd agente-csv-gemini

# 2. Criar ambiente virtual
python -m venv .venv
source .venv/Scripts/activate  # (Windows)
# ou
source .venv/bin/activate      # (Linux/Mac)

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar o app
streamlit run app.py
```

---

## 🌐 Versão Online (Demo)

🎯 [Acesse o agente no Streamlit](https://agente-csv-gemini.streamlit.app/)  
📦 [Repositório no GitHub](https://github.com/Abnadablys/agente-csv-gemini)

---

## 🧾 Autor

**Abnadablys Sinclair**  
📍 Projeto desenvolvido para a disciplina de **Agentes Autônomos e Inteligência Artificial**.  
📅 2025
