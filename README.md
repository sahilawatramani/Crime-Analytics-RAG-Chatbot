# 🚨 Crime Analytics RAG Chatbot: India (2001–2014)

> **AI-Powered Analysis of Crimes Against Women in India**  
> _Explore and analyze official crime data with natural language queries, interactive filters, and robust analytics._

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Data Sources](#data-sources)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## 📝 Overview

**Crime Analytics RAG Chatbot** is an open-source web application that empowers users to explore and analyze crimes against women in India from 2001 to 2014. Leveraging Retrieval-Augmented Generation (RAG) and natural language processing, the chatbot makes complex crime data accessible to everyone—researchers, journalists, policymakers, and the public.

The system supports both free-form natural language queries and structured filter-based exploration, providing instant insights, breakdowns, and trends from official government datasets.

---

## ✨ Features

- **Natural Language Querying:**  
  Ask questions like “Show me dowry deaths in Bihar from 2005 to 2010” and get instant, human-readable answers.
- **Interactive Filters:**  
  Drill down by year, state/UT, and crime type with intuitive controls.
- **Robust Analytics Engine:**  
  Backend powered by Python, Pandas, and custom analytics for accurate, fast results.
- **Open Source & Extensible:**  
  Transparent, modular, and community-driven codebase.
- **Secure & Reliable:**  
  Input validation, logging, and robust error handling throughout.

---

## 🏗️ System Architecture

**Frontend:**
- HTML, CSS, JavaScript
- Responsive UI for chat and filters

**Backend:**
- Python, Flask web server
- Pandas for data processing and analytics
- Custom modules for query parsing, parameter extraction, and statistical analysis
- RESTful API endpoints for queries, stats, and filter options

**Data:**
- Official government crime statistics (2001–2014)
- Preprocessed and normalized for consistency and accuracy

**NLP/RAG:**
- [If applicable: spaCy, NLTK, or custom rule-based query analyzer for intent and parameter extraction]
- Retrieval-Augmented Generation for context-aware responses (if using LLMs or embeddings)


---

## 📊 Data Sources

- **Primary:** https://www.kaggle.com/datasets/greeshmagirish/crime-against-women-20012014-india
- **Preprocessing:** Data cleaned, normalized, and validated for consistency and accuracy
- **Coverage:**  
  - Years: 2001–2014  
  - States/UTs: All official Indian states and union territories  
  - Crime Types: Rape, Dowry Deaths, Kidnapping, Assault, Cruelty by Husband/Relatives, and more

---

## ⚡ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sahilawatramani/Crime-Analytics-RAG-Chatbot.git
cd Crime-Analytics-RAG-Chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python flask_app.py
```
---

## 💡 Usage

- **Ask a Question:**  
  Type a natural language query (e.g., “Show me rape cases in Delhi from 2010 to 2012”) or use the filters to select years, states, and crime types.
- **View Results:**  
  Get instant summaries.
- **Explore Further:**  
  Refine your query or filters to dig deeper into the data.
- **Sample Queries:**  
  - “Compare dowry deaths across states in 2005”
  - “Show trend of assault cases in Maharashtra from 2001 to 2014”
  - “Which state had the highest number of kidnapping cases in 2010?”

---

## 🧪 Testing

- **Unit Tests:**  
  Run backend tests with:
  ```bash
  python -m unittest discover
  ```
- **Manual Testing:**  
  - Try various queries and filter combinations in the UI.
  - Check for correct results, error handling, and performance.

---

## 🤝 Contributing

We welcome contributions!  
- Open issues for bugs, feature requests, or questions.
- Submit pull requests for improvements.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 If you find this project useful, please ⭐ star the repo and share your feedback!
