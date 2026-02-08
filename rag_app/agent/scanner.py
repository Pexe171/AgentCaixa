import os
import sys

# Garante que o Python ache as pastas do projeto
sys.path.append(os.getcwd())

from rag_app.agent.vector_index import VectorIndex

def force_ingest():
    # Caminho onde os arquivos estão
    target_dir = os.path.join(os.getcwd(), "data", "knowledge")
    
    print(f"--- INICIANDO VARREDURA FORÇADA ---")
    print(f"Verificando pasta: {target_dir}")

    if not os.path.exists(target_dir):
        print("ERRO: Pasta data/knowledge nao encontrada!")
        return

    vi = VectorIndex()
    arquivos = os.listdir(target_dir)
    print(f"Arquivos encontrados na pasta: {arquivos}")

    for arq in arquivos:
        # Forçamos a leitura de PDF e DOC (mesmo que o DOC possa falhar, queremos ver o log)
        if arq.lower().endswith(('.pdf', '.doc', '.docx')):
            caminho_completo = os.path.join(target_dir, arq)
            tamanho = os.path.getsize(caminho_completo)
            
            if tamanho > 0:
                print(f">>> TENTANDO INDEXAR: {arq} ({tamanho} bytes)")
                try:
                    vi.ingest_file(caminho_completo)
                    print(f"✅ SUCESSO: {arq}")
                except Exception as e:
                    print(f"❌ ERRO ao ler {arq}: {e}")
            else:
                print(f"⚠️ PULANDO: {arq} esta vazio.")
        else:
            print(f"ℹ️ IGNORANDO: {arq} (extensao nao suportada)")

if __name__ == "__main__":
    force_ingest()
