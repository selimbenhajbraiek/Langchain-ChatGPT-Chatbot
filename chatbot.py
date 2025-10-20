from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

api_key = "YOUR_OPENAI_API_KEY"  # replace with your key

# 1 Load webpage
url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
loader = WebBaseLoader(url)
raw_documents = loader.load()

# 2 Split the webpage text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)

# 3️ Convert text chunks into embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# 4️ Store embeddings in a FAISS vector database
vectorstore = FAISS.from_documents(documents, embeddings)

# 5️ Create memory for the conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 6️ Create a conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0),
    vectorstore.as_retriever(),
    memory=memory
)

# 7️ Ask a question about the webpage
query = "Who is considered the father of Artificial Intelligence?"
result = qa({"question": query})

print("Question:", query)
print("Answer:", result["answer"])
