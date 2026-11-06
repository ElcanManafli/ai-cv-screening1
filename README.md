# 🚀 TechConnect: AI-Powered CV Screening & Offer Generator

TechConnect is an intelligent, automated recruitment assistant built for hackathons. It streamlines the hiring process by parsing multiple CVs in parallel, evaluating candidates against a job description using Google Gemini AI, ranking them by match percentage, and automatically generating customized offer letters.

---

## ✨ Key Features

- **Parallel CV Processing:** Upload multiple PDF CVs simultaneously and analyze them efficiently using concurrent processing (`ThreadPoolExecutor`).
- **AI-Driven Evaluation:** Powered by **Google Gemini API (`gemini-1.5-flash`)** to extract candidate details, calculate match percentages, list key strengths, and identify missing skills.
- **Automated Offer Letter Generation:** Instantly generate professional, customized offer letters for top-tier candidates.
- **Secure & Seamless UI:** Built with **Streamlit** for a clean, intuitive, and lightning-fast user experience with secure backend API key management.

---

## 🛠️ Tech Stack

- **Frontend & UI:** Streamlit
- **AI / LLM:** Google Generative AI (`google-generativeai` SDK, Gemini 1.5 Flash)
- **PDF Processing:** `pypdf` / `pdfplumber`
- **Language:** Python 3.9+

---

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
   cd YOUR_REPOSITORY
