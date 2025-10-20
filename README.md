# LangChain Chatbot for Website Content

## 🚀 Project Overview

This project demonstrates how to build a **custom AI chatbot** using **LangChain**, **OpenAI embeddings**, and **FAISS** that can answer questions based on the content of a website.  

💡 **Problem:**  
ChatGPT cannot answer questions about your private company or website because it only knows publicly available data.  

✅ **Solution:**  
This project fetches your website content, converts it into embeddings, stores them in a vector database, and allows ChatGPT to answer questions based on your own content using a **conversational retrieval chain**.

---

## 🛠 Features

- Load and parse web pages using `WebBaseLoader`.
- Split long documents into smaller chunks with `RecursiveCharacterTextSplitter`.
- Convert text chunks into **semantic embeddings** using OpenAI.
- Store embeddings in a **FAISS vector store** for fast semantic search.
- Build a **conversational retrieval chain** with memory for multi-turn Q&A.
- Ask questions about your website and get accurate, context-aware answers.

---

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/langchain-web-chatbot.git
cd langchain-web-chatbot
pip install -r requirements.txt
