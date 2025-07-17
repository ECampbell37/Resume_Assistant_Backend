# 🔧 Resume Assistant – Backend

This is the core system that powers the Resume Assistant app. It handles reading resumes, giving AI feedback, matching job roles, and answering questions with a chatbot.

It's designed to run efficiently in **Docker** containers and be easily deployed on **AWS ECS Fargate**. It's protected by an **Application Load Balancer** and **Cloudflare** for security and smooth operation. Plus, updates are automatically deployed using **GitHub Actions**.

---

## 📦 What it Does

* **Processes PDF Resumes:** Take and read information from your PDF resumes.
* **Generates AI Feedback:** Uses AI to extract text and provide helpful feedback on your resume.
* **Suggests Job Roles:** Recommends suitable job roles based on your resume content.
* **Smart Chatbot:** A chatbot that understands your resume and offers personalized career advice. 

---

## 🛠 Technologies Used

* **Framework:** FastAPI
* **AI:** LangChain and OpenAI
* **Deployment:** Docker, ECS Fargate, ALB
* **Infrastructure:** AWS and Cloudflare (for cloud services and security)
* **Monitoring:** CloudWatch

---

## ⚙️ How to Develop Locally

To get started with local development:

1.  **Clone the project:**
    ```bash
    git clone [https://github.com/your-username/resume-assistant-backend](https://github.com/your-username/resume-assistant-backend)
    cd resume-assistant-backend
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up your OpenAI API Key:**
    Create a file named `.env` and add your OpenAI API key:
    ```
    OPENAI_API_KEY=
    ```
4.  **Start the development server:**
    ```bash
    uvicorn main:app --reload --port 80
    ```

---

## 🐳 How to Use with Docker

To build and run the application using Docker:

1.  **Build the Docker image:**
    ```bash
    docker build -t resume-api .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -p 80:80 resume-api
    ```

---

## 🌐 Available API Endpoints

Here are the main ways you can interact with the system:

| Method | Path        | What it Does                    |
| :----- | :---------- | :------------------------------ |
| `POST` | `/analyze`  | Uploads and analyzes a resume   |
| `POST` | `/chat`     | Interacts with the resume chatbot |
| `POST` | `/jobmatch` | Suggests relevant job roles     |
| `GET`  | `/health`   | Checks if the system is running |

---

## 📁 Project Layout

Here's a simple overview of the project's main files:

* `main.py`: The main starting point for the FastAPI application.
* `analysis.py`: Contains the logic for scoring resumes.
* `job_match.py`: Handles the recommendations for job roles.
* `chatbot.py`: Manages the chatbot and user conversation history.
* `Dockerfile`: Instructions for building the Docker container.

---

## 🔗 Related Projects

* [Frontend Repository](https://github.com/ECampbell37/Resume_Assistant_Frontend): The user interface for the Resume Assistant, built with Next.js and Supabase.
* [Backend Repository](https://github.com/ECampbell37/Resume_Assistant_Backend): This very project!

---

## 👤 Author

Elijah Campbell-Ihim
[elijahcampbellihimportfolio.com](https://elijahcampbellihimportfolio.com)
