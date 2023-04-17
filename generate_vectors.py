import os
import pickle
import time
from datetime import datetime
from pytz import timezone
from openai.error import RateLimitError
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

os.environ['OPENAI_API_KEY'] = ""
EMBED = 'openai'
embedding_scheme = OpenAIEmbeddings()

def create_vectorstores(kb_dir='knowledgebase'):
    for pages_path in os.listdir(kb_dir):
        source = os.path.splitext(pages_path)[0]
        out_path = f"vectorstore/{EMBED.lower()}_embeddings/{source}.pkl"
        if os.path.exists(out_path):
            continue
            
        pages = pickle.load(open(os.path.join(kb_dir, pages_path), 'rb'))
        index = FAISS.from_documents([pages.pop(0)], embedding_scheme)

        i, step = 0, 30
        while i<len(pages):
            _d = datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{_d}   Processing pages {i}:{i+step}")
            texts = [d.page_content for d in pages[i:i+step]]
            meta = [d.metadata for d in pages[i:i+step]]
            try:
                index.add_texts(texts, meta)
                i += step
            except RateLimitError:
                print("Hit RateLimit @ i=",i)
                time.sleep(60)
            except ConnectionResetError:
                print("Connection was reset")
                time.sleep(10)
        pickle.dump(index, open(out_path, "wb"))


if __name__ == "__main__":
    create_vectorstores()
