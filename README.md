## GPT3-powered Question Answering for PyTorch

### WHAT
A Q&A bot that tries to answer PyTorch queries by looking up the API docs, blog posts on pytorch.org, and solved forum threads. The bot responds with an answer along with the documents it sourced its answer from.

### How does this work
The bot does mainly 2 things: i) finding the most relevant documents that might contain the answer to the question, and ii) constructing a succinct response to the query from the relevant documents. 

For more details: https://subramen.github.io/garden/posts/ai-searcher-2/