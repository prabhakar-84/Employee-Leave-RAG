import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# ==========================================
# ENVIRONMENT
# ==========================================

load_dotenv()


# ==========================================
# CONFIGURATION
# ==========================================

PDF_PATH = "data/Employee Leave & Attendance Policy-.pdf"
CHROMA_PATH = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"


# ==========================================
# STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="Employee Leave & Attendance Assistant",
    page_icon="📋",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #f5f7fb;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📋 Policy Assistant")

    st.write(
        "AI-powered chatbot for answering "
        "Employee Leave & Attendance Policy questions."
    )

    st.divider()

    st.subheader("📌 Policy Coverage")

    st.markdown(
        """
        - Earned Leave (EL)
        - Casual Leave (CL)
        - Sick Leave (SL)
        - Optional Holiday (OH)
        - Maternity Leave
        - Paternity Leave
        - Bereavement Leave
        - Compensatory Off
        - Leave Without Pay
        """
    )

    st.divider()

    st.subheader("👥 Employee Records")

    st.markdown(
        """
        - Aarav Singh
        - Priya Nair
        - Rohan Kumar
        - Sneha Rao
        - Arjun Patel
        - Rahul Mehta
        """
    )

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.selected_question = None

        st.rerun()

    st.divider()

    st.caption("🔎 Powered by RAG")
    st.caption("📄 Employee Leave & Attendance Policy")


# ==========================================
# MAIN HEADER
# ==========================================

st.markdown(
    '<div class="main-title">'
    '📋 Employee Leave & Attendance Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about employee leave, attendance, '
    'eligibility and policy rules.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# GROQ API KEY
# ==========================================

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:

    st.error(
        "GROQ_API_KEY not found in .env file."
    )

    st.stop()


# ==========================================
# LOAD PDF DOCUMENTS
# ==========================================

@st.cache_resource
def load_policy_documents():

    if not os.path.exists(PDF_PATH):

        st.error(
            f"Policy PDF not found: {PDF_PATH}"
        )

        st.stop()

    try:

        loader = PyPDFLoader(PDF_PATH)

        documents = loader.load()

        if not documents:

            st.error(
                "The policy PDF is empty or could not be read."
            )

            st.stop()

        return documents

    except Exception as e:

        st.error(
            "Unable to load the policy PDF."
        )

        st.error(
            f"PDF error: {e}"
        )

        st.stop()


# ==========================================
# CREATE / LOAD VECTOR DATABASE
# ==========================================

@st.cache_resource
def create_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    if os.path.exists(CHROMA_PATH):

        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

    else:

        documents = load_policy_documents()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_documents(
            documents
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )

    return vectorstore


# ==========================================
# INITIALIZE VECTOR DATABASE
# ==========================================

vectorstore = create_vectorstore()


# ==========================================
# RETRIEVER
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 6
    }
)


# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model=GROQ_MODEL,
    groq_api_key=groq_api_key,
    temperature=0
)


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "selected_question" not in st.session_state:

    st.session_state.selected_question = None


# ==========================================
# EMPLOYEE RECORDS
# ==========================================

employee_names = [
    "Rahul Mehta",
    "Aarav Singh",
    "Priya Nair",
    "Rohan Kumar",
    "Sneha Rao",
    "Arjun Patel"
]


# ==========================================
# EXACT EMPLOYEE LEAVE BALANCES
# ==========================================

employee_balances = {

    "Aarav Singh": {
        "EL": "14 days",
        "CL": "5 days",
        "SL": "8 days"
    },

    "Priya Nair": {
        "EL": "9 days",
        "CL": "4 days",
        "SL": "7 days"
    },

    "Rohan Kumar": {
        "EL": "21 days",
        "CL": "6 days",
        "SL": "9 days"
    },

    "Sneha Rao": {
        "EL": "12 days",
        "CL": "2 days",
        "SL": "6 days"
    },

    "Arjun Patel": {
        "EL": "30 days",
        "CL": "7 days",
        "SL": "10 days"
    },

    "Rahul Mehta": {
        "EL": "12.5 days",
        "CL": "5 days",
        "SL": "8 days"
    }
}


# ==========================================
# NORMALIZE TEXT
# ==========================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        "",
        text
    )

    return text


# ==========================================
# FIND EMPLOYEE DOCUMENTS
# ==========================================

def find_employee_documents(
    question,
    documents
):

    question_normalized = normalize_text(
        question
    )

    employee_documents = []

    for employee_name in employee_names:

        name_normalized = normalize_text(
            employee_name
        )

        if name_normalized in question_normalized:

            for document in documents:

                page_normalized = normalize_text(
                    document.page_content
                )

                if name_normalized in page_normalized:

                    employee_documents.append(
                        document
                    )

            break

    return employee_documents


# ==========================================
# FIND EMPLOYEE NAME
# ==========================================

def find_employee_name(question):

    question_normalized = normalize_text(
        question
    )

    for employee_name in employee_balances:

        name_normalized = normalize_text(
            employee_name
        )

        if name_normalized in question_normalized:

            return employee_name

    return None


# ==========================================
# FIND LEAVE TYPE
# ==========================================

def find_leave_type(question):

    question_lower = question.lower()

    # Earned Leave
    if re.search(
        r"\bearned\s+leave\b",
        question_lower
    ) or re.search(
        r"\bel\b",
        question_lower
    ):

        return "EL"

    # Casual Leave
    if re.search(
        r"\bcasual\s+leave\b",
        question_lower
    ) or re.search(
        r"\bcl\b",
        question_lower
    ):

        return "CL"

    # Sick Leave
    if re.search(
        r"\bsick\s+leave\b",
        question_lower
    ) or re.search(
        r"\bsl\b",
        question_lower
    ):

        return "SL"

    return None


# ==========================================
# CHECK BALANCE QUESTION
# ==========================================

def is_balance_question(question):

    question_lower = question.lower()

    balance_keywords = [
        "balance",
        "remaining",
        "remain",
        "available",
        "left",
        "how much leave do",
        "how many leave do",
        "how many days do"
    ]

    applied_keywords = [
        "apply",
        "applied",
        "applying",
        "request",
        "requested",
        "requesting",
        "took",
        "taken",
        "used",
        "use",
        "approved leave"
    ]

    has_balance_keyword = any(
        keyword in question_lower
        for keyword in balance_keywords
    )

    has_applied_keyword = any(
        keyword in question_lower
        for keyword in applied_keywords
    )

    if has_applied_keyword:
        return False

    return has_balance_keyword


# ==========================================
# SUGGESTED QUESTIONS
# ==========================================

if not st.session_state.messages:

    st.markdown(
        '<div class="info-box">'
        '<b>💡 Try asking one of these questions</b>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "📅 How many EL days are available?",
            use_container_width=True
        ):

            st.session_state.selected_question = (
                "How many Earned Leave days are available per year?"
            )

            st.rerun()

        if st.button(
            "🏖️ How many CL days are available?",
            use_container_width=True
        ):

            st.session_state.selected_question = (
                "How many Casual Leave days are available per year?"
            )

            st.rerun()

        if st.button(
            "🤒 What is the Sick Leave policy?",
            use_container_width=True
        ):

            st.session_state.selected_question = (
                "What is the Sick Leave policy?"
            )

            st.rerun()

    with col2:

        if st.button(
            "👶 How many Paternity Leave days?",
            use_container_width=True
        ):

            st.session_state.selected_question = (
                "How many Paternity Leave days are available?"
            )

            st.rerun()

        if st.button(
            "📚 What is the EL carry-forward limit?",
            use_container_width=True
        ):

            st.session_state.selected_question = (
                "What is the Earned Leave carry-forward limit?"
            )

            st.rerun()

        if st.button(
            "📝 How do I apply for leave?",
            use_container_width=True
        ):

            st.session_state.selected_question = (
                "How do I apply for leave?"
            )

            st.rerun()


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            sources = message.get(
                "sources",
                []
            )

            if sources:

                st.caption(
                    "📚 Sources: "
                    + ", ".join(sources)
                )


# ==========================================
# CHAT INPUT
# ==========================================

chat_question = st.chat_input(
    "Ask about leave policy...",
    key="main_chat_input"
)

question = chat_question

if (
    not question
    and st.session_state.selected_question
):

    question = st.session_state.selected_question

    st.session_state.selected_question = None


# ==========================================
# PROCESS QUESTION
# ==========================================

if question:

    # ======================================
    # CHECK EXACT EMPLOYEE BALANCE
    # ======================================

    employee_match = find_employee_name(
        question
    )

    leave_type = find_leave_type(
        question
    )


    # ======================================
    # EXACT EMPLOYEE BALANCE RESPONSE
    # ======================================

    if (
        employee_match
        and leave_type
        and is_balance_question(question)
    ):

        answer = (
            f"{employee_match}'s "
            f"{leave_type} balance is "
            f"{employee_balances[employee_match][leave_type]}."
        )


        # ======================================
        # FIND ACTUAL EMPLOYEE SOURCE PAGE
        # ======================================

        policy_documents = load_policy_documents()

        employee_docs = find_employee_documents(
            question,
            policy_documents
        )

        source_pages = []

        for document in employee_docs:

            page_number = document.metadata.get(
                "page"
            )

            if page_number is not None:

                source = (
                    f"PDF Page {page_number + 1}"
                )

                if source not in source_pages:

                    source_pages.append(
                        source
                    )


        # ======================================
        # FALLBACK SOURCE
        # ======================================

        if not source_pages:

            source_pages = [
                "Policy PDF"
            ]


        # ======================================
        # USER MESSAGE
        # ======================================

        with st.chat_message("user"):

            st.markdown(
                question
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # ======================================
        # ASSISTANT MESSAGE
        # ======================================

        with st.chat_message("assistant"):

            st.markdown(
                answer
            )

            st.caption(
                "📚 Sources: "
                + ", ".join(source_pages)
            )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_pages
            }
        )

        st.stop()


    # ======================================
    # NORMAL RAG QUESTION
    # ======================================

    with st.chat_message("user"):

        st.markdown(
            question
        )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # ======================================
    # ASSISTANT
    # ======================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching policy..."
        ):

            # ----------------------------------
            # SEMANTIC SEARCH
            # ----------------------------------

            try:

                retrieved_docs = retriever.invoke(
                    question
                )

            except Exception as e:

                st.error(
                    f"Search error: {e}"
                )

                retrieved_docs = []


            # ----------------------------------
            # EMPLOYEE SEARCH
            # ----------------------------------

            policy_documents = load_policy_documents()

            employee_docs = find_employee_documents(
                question,
                policy_documents
            )


            # ----------------------------------
            # PRIORITIZE EMPLOYEE DOCUMENTS
            # ----------------------------------

            if employee_docs:

                retrieved_docs = employee_docs


            # ----------------------------------
            # REMOVE DUPLICATES
            # ----------------------------------

            unique_docs = []

            seen_content = set()

            for document in retrieved_docs:

                content = (
                    document.page_content.strip()
                )

                if content not in seen_content:

                    seen_content.add(
                        content
                    )

                    unique_docs.append(
                        document
                    )

            retrieved_docs = unique_docs


            # ----------------------------------
            # NO SEARCH RESULTS
            # ----------------------------------

            if not retrieved_docs:

                answer = (
                    "I could not find this information "
                    "in the employee leave policy."
                )

                source_pages = []

            else:

                # ----------------------------------
                # CREATE CONTEXT
                # ----------------------------------

                context = "\n\n".join(
                    document.page_content
                    for document in retrieved_docs
                )


                # ----------------------------------
                # SOURCE PAGES
                # ----------------------------------

                source_pages = []


                # ----------------------------------
                # PROMPT
                # ----------------------------------

                prompt = f"""
You are an Employee Leave & Attendance Policy assistant.

Answer the user's question using ONLY the provided policy context.

Rules:

1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer is not available in the context, say:
"I could not find this information in the employee leave policy."
4. Give a clear and simple answer.
5. Mention important conditions or restrictions when relevant.
6. If the question asks about an employee's leave balance,
use the employee record from the context.
7. Do not confuse one employee's balance with another employee's balance.
8. If an employee name is mentioned, answer only for that employee.
9. Keep the answer professional and concise.

Policy Context:

{context}

User Question:

{question}

Answer:
"""


                # ----------------------------------
                # LLM RESPONSE
                # ----------------------------------

                try:

                    response = llm.invoke(
                        prompt
                    )

                    answer = response.content


                    # ==================================
                    # RELEVANT SOURCE PAGE SELECTION
                    # ==================================

                    answer_text = answer.lower()

                    # Remove common words that do not help
                    # identify the supporting page.
                    stop_words = {
                        "the",
                        "and",
                        "are",
                        "is",
                        "was",
                        "were",
                        "for",
                        "from",
                        "with",
                        "this",
                        "that",
                        "what",
                        "how",
                        "many",
                        "days",
                        "day",
                        "per",
                        "year",
                        "leave",
                        "policy",
                        "available",
                        "employee"
                    }


                    # Words from the user's question
                    question_words = re.findall(
                        r"\b[a-zA-Z0-9]+\b",
                        question.lower()
                    )

                    question_words = [
                        word
                        for word in question_words
                        if len(word) > 2
                        and word not in stop_words
                    ]


                    # Words from the answer
                    answer_words = re.findall(
                        r"\b[a-zA-Z0-9]+\b",
                        answer_text
                    )

                    answer_words = [
                        word
                        for word in answer_words
                        if len(word) > 2
                        and word not in stop_words
                    ]


                    # Numbers in answer
                    answer_numbers = re.findall(
                        r"\b\d+(?:\.\d+)?\b",
                        answer_text
                    )


                    scored_sources = []


                    for document in retrieved_docs:

                        page_number = (
                            document.metadata.get(
                                "page"
                            )
                        )

                        if page_number is None:
                            continue


                        page_text = (
                            document.page_content.lower()
                        )


                        score = 0


                        # ----------------------------------
                        # QUESTION WORD MATCH
                        # ----------------------------------

                        for word in question_words:

                            if word in page_text:

                                score += 3


                        # ----------------------------------
                        # ANSWER WORD MATCH
                        # ----------------------------------

                        for word in answer_words:

                            if word in page_text:

                                score += 2


                        # ----------------------------------
                        # ANSWER NUMBER MATCH
                        # ----------------------------------

                        for number in answer_numbers:

                            if number in page_text:

                                score += 6


                        scored_sources.append(
                            (
                                score,
                                page_number
                            )
                        )


                    # ----------------------------------
                    # SORT BY SUPPORT SCORE
                    # ----------------------------------

                    scored_sources.sort(
                        key=lambda item: item[0],
                        reverse=True
                    )


                    # ----------------------------------
                    # SELECT ONLY STRONGEST SOURCE
                    # ----------------------------------

                    if scored_sources:

                        best_score = (
                            scored_sources[0][0]
                        )

                        if best_score > 0:

                            best_page = (
                                scored_sources[0][1]
                            )

                            source_pages = [
                                f"PDF Page {best_page + 1}"
                            ]


                    # ----------------------------------
                    # FALLBACK
                    # ----------------------------------

                    if not source_pages:

                        if retrieved_docs:

                            page_number = (
                                retrieved_docs[0]
                                .metadata
                                .get("page")
                            )

                            if page_number is not None:

                                source_pages = [
                                    f"PDF Page {page_number + 1}"
                                ]

                            else:

                                source_pages = [
                                    "Policy PDF"
                                ]

                        else:

                            source_pages = [
                                "Policy PDF"
                            ]


                except Exception as e:

                    answer = (
                        "Sorry, I could not process "
                        "your question right now. "
                        "Please try again."
                    )

                    st.error(
                        f"AI response error: {e}"
                    )


        # ======================================
        # SHOW ANSWER
        # ======================================

        st.markdown(
            answer
        )


        # ======================================
        # SHOW SOURCES
        # ======================================

        if source_pages:

            st.caption(
                "📚 Sources: "
                + ", ".join(source_pages)
            )


    # ======================================
    # SAVE ASSISTANT MESSAGE
    # ======================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_pages
        }
    )
