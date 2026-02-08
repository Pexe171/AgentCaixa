import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorIndex:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.index_path = os.path.join(os.getcwd(), "data", "index")

    def ingest_file(self, file_path: str):
        print(f"📄 Indexando Procedimento: {os.path.basename(file_path)}")
        
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        
        # Aumentamos o chunk_size para 1500 para garantir que o passo a passo 
        # do APP FGTS caiba inteiro em um único bloco de memória da IA.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=200,
            separators=["\n\n", "\n", "▪", "•", "."] # Adicionado foco nos marcadores
        )
        docs = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        
        os.makedirs(self.index_path, exist_ok=True)
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            db = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            db.merge_from(vectorstore)
            db.save_local(self.index_path)
        else:
            vectorstore.save_local(self.index_path)
        print(f"✅ Memória operacional atualizada!")
