# 🔧 Resume Assistant – Backend

FastAPI backend that powers resume parsing, AI feedback, job role matching, and chatbot Q&A for the Resume Assistant app.

Designed for containerized deployment using **Docker** and **AWS ECS Fargate**, with a public API served behind an **Application Load Balancer**, secured via **Cloudflare**, and deployed automatically via **GitHub Actions**.

---

## 📦 Features

- 📄 Accepts and processes PDF resumes
- 🧠 Extracts text and generates AI feedback
- 🎯 Recommends job roles based on resume content
- 💬 Resume-aware chatbot with memory per user
- ⚙️ Health check endpoint for load balancing
- 🐳 Production-ready Docker build using Gunicorn + UvicornWorker

---

## 🛠 Stack

- **Framework:** FastAPI
- **AI Engine:** LangChain + OpenAI
- **Deployment:** Docker, ECS Fargate, ALB
- **Infra:** AWS + Cloudflare
- **Monitoring:** CloudWatch

---

## ⚙️ Local Development

```bash
git clone https://github.com/your-username/resume-assistant-backend
cd resume-assistant-backend
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=
```

Start the development server:

```bash
uvicorn main:app --reload --port 8000
```

---

## 🐳 Docker Usage

```bash
docker build -t resume-api .
docker run -p 80:80 resume-api
```

---

## 🌐 API Endpoints

| Method | Path        | Description               |
|--------|-------------|---------------------------|
| POST   | `/analyze`  | Analyze uploaded resume   |
| POST   | `/chat`     | Resume-aware chatbot Q&A  |
| POST   | `/match`    | Suggest relevant job roles|
| GET    | `/health`   | Health check for ALB      |

---

## 📁 Project Structure

```
main.py          # FastAPI entry point
analysis.py      # Resume scoring logic
job_match.py     # Role recommendation logic
chatbot.py       # Chatbot + user memory
Dockerfile       # Container build config
```

---

## 🔗 Related Repositories

- [Frontend Repository](https://github.com/ECampbell37/Resume_Assistant_Frontend) – Next.js + Supabase client UI
- [Backend Repository](https://github.com/ECampbell37/Resume_Assistant_Backend) – This repo

---

## 👤 Author

Elijah Campbell-Ihim  
[elijahcampbellihimportfolio.com](https://elijahcampbellihimportfolio.com)
