# 🏍️ DataEngineerGPT Chat Bot

**DataEngineerGPT** is a highly specialized Streamlit chatbot powered by advanced LLMs hosted on Groq Cloud. Designed for data engineering, DevOps, and cloud architecture support, this assistant provides real-time, context-aware help with building and debugging production-grade data systems.

---

## 🚀 Features

* ✅ Interactive Streamlit chat interface
* 🤖 Backed by state-of-the-art LLMs:

  * `meta-llama/llama-4-scout-17b-16e-instruct`
  * `meta-llama/llama-4-maverick-17b-128e-instruct`
  * `qwen-qwq-32b`
  * `deepseek-r1-distill-llama-70b`
  * `compound-beta`
* 🧠 System prompt tuned for Data Engineering expertise
* ⚡ Fast responses powered by Groq's ultra-performant API
* 🔐 Secure configuration via Streamlit secrets

---

## 🧰 Tech Stack

* [Streamlit](https://streamlit.io/)
* [Groq API](https://console.groq.com/)
* Python 3.10+
* Modern LLMs (LLaMA 4, Qwen, DeepSeek, etc.)

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/lostdir/DATAENG-GPT.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Secrets

Create a file at `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
.
├── app.py                 # Streamlit chatbot app
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── secrets.toml       # Groq API key
├── .gitignore
└── README.md
```

---

## 📌 Example Use Cases

* Get code for streaming pipelines, CDC, and ETL
* Debug Spark, Kafka, Airflow or SQL queries
* Generate infrastructure plans (Terraform, CI/CD)
* Ask anything about Data Engineering best practices

---

## 👨‍💼 Author

🔗 [LinkedIn](https://linkedin.com/in/harshalkh192) • [GitHub](https://github.com/lostdir)

---

## 📝 License

MIT License – use, modify, and share freely.
