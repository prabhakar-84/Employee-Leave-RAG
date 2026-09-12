# Employee Leave & Attendance Policy RAG Chatbot

## Project Overview

Employee Leave & Attendance Policy RAG Chatbot is an AI-powered application that allows employees to ask questions about company leave and attendance policies.

The chatbot uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the company's Employee Leave & Attendance Policy PDF and generate accurate answers.

## Key Features

- Ask questions about employee leave policies
- Retrieve employee-specific leave balances
- Supports Earned Leave (EL), Casual Leave (CL), and Sick Leave (SL)
- Provides policy information such as leave entitlement and carry-forward limits
- Shows PDF source pages for answers
- Handles questions where information is not available in the policy
- Interactive chat interface using Streamlit
- Local HuggingFace embeddings
- Groq LLM for answer generation

## Technologies Used

- Python
- Streamlit
- LangChain
- LangChain Community
- ChromaDB
- HuggingFace Sentence Transformers
- Groq
- PyPDF
- python-dotenv

## RAG Architecture

```text
PDF Policy
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
HuggingFace Embeddings
    ↓
Chroma Vector Database
    ↓
User Question
    ↓
Relevant Document Retrieval
    ↓
Groq LLM
    ↓
Final Answer + Source Page
```

## Project Structure

```text
Employee-Leave-RAG/
│
├── data/
│   └── Employee Leave & Attendance Policy-.pdf
│
├── chroma_db/
│   └── Vector database files
│
├── app.py
├── requirements.txt
├── .env
├── README.md
└── venv/
```

## Run the Application

Open the terminal in the `Employee-Leave-RAG` project folder and run:

```powershell
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Sample Questions

You can ask questions such as:

- How many Earned Leave days are available per year?
- How many Casual Leave days are available?
- What is the Sick Leave policy?
- How many days of Paternity Leave are provided?
- What is the maximum Earned Leave carry-forward limit?
- What is Rahul Mehta's Earned Leave balance?
- What is Arjun Patel's Sick Leave balance?
- How do I apply for leave?

## Installation

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not upload the `.env` file or API key to GitHub.

## Testing

The chatbot was tested with:

- Employee-specific leave balance questions
- General leave policy questions
- Leave application process questions
- Leave carry-forward questions
- Questions with unavailable information

The chatbot provides relevant answers along with PDF source pages.

## Disclaimer

This project is developed for educational and demonstration purposes.

The employee data and company policy used in this project are fictional sample data and should not be considered as an actual company policy.

## Author

Developed as an AI and RAG-based project for learning and demonstration purposes.