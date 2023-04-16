import os
import pickle

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio.platform.flask import webio_view
from flask import Flask

application = Flask(__name__)
os.environ['OPENAI_API_KEY'] = "<api-key>"
base_llm = OpenAI(temperature=0.2)
qa_chain = load_qa_with_sources_chain(base_llm, chain_type="stuff")

def display_output(response):
    for msg in response["message"]:
        refs = "\n".join(msg["citations"])
        response = f"""
## Answer derived from {msg['source']}:
{msg['answer']}

### References:
{refs}
        """
        put_markdown(response)


def remap_sources(sources):
    LOOKUP = {"Blog": "blog", "API Docs & Tutorials": "docs", "User Forum": "forum"}
    return list(map(LOOKUP.get, sources))


def get_answer():
    response = {"message": []}
    question = pin["question"]
    sources = pin["sources"]
    src_files = remap_sources(sources)
    print(f"processing Q: {question} from src: {sources}")
    for src, file in zip(sources, src_files):
        vectordb_path = f"vectorstore/openai_embeddings/{file}.pkl"
        db = pickle.load(open(vectordb_path, "rb"))
        print(f"[src] loaded db")
        similar_docs = db.similarity_search(question, k=4)
        print(f"[src] simsearch done")
        gpt_response = qa_chain.run(input_documents=similar_docs, question=question)
        print(f"[src] gpt has responded")
        print(gpt_response)
        answer, links = gpt_response.split("SOURCES:")
        answer = answer.strip(" ").strip("\n")
        citations = [
            l.strip(" ").strip("\n").strip(",") for l in links.split(",") if l != " "
        ]
        response["message"].append(
            {"source": src, "answer": answer, "citations": citations}
        )
    display_output(response)

def display_form():
    """
    PyTorch Answer Engine

    """
    put_info("PyTorch Answer Engine")
    put_input("question", label="Your PyTorch question: ", type=TEXT)
    put_text("\n")
    put_checkbox(
        "sources",
        label="Search across",
        options=["Blog", "API Docs & Tutorials", "User Forum"],
        inline=False,
        value=["API Docs & Tutorials"],
    )
    put_text("\n")
    put_button(label="submit", onclick=get_answer)

application.add_url_rule('/', 'webio_view', webio_view(display_form), methods=['GET', 'POST', 'OPTIONS'])

if __name__ == "__main__":
    application.run(host='0.0.0.0')