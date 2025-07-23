
import tiktoken
import pinecone
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import CohereEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import (ChatPromptTemplate,SystemMessagePromptTemplate,)


urls = [
"https://www.arabnews.com/category/tags/riyadh",
"https://www.akhbaar24.com/",
"https://www.timeanddate.com/weather/saudi-arabia/riyadh/ext"]

loader = WebBaseLoader(urls)
data = loader.load()

# ------------------------------------------------------------------------------------------------

encoding_name = tiktoken.get_encoding("cl100k_base")
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# ------------------------------------------------------------------------------------------------


text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=25)
docs = text_splitter.split_documents(data)


# ------------------------------------------------------------------------------------------------


embeddings = CohereEmbeddings(model='embed-multilingual-v3.0',# for english (embed-english-v3.0)
                              cohere_api_key='COHERE_API_KEY')

llm = ChatOpenAI(openai_api_key = "OPENAI_API_KEY"
                 ,max_tokens=800, temperature=0.7, verbose=True)

# ------------------------------------------------------------------------------------------------


pinecone.init(
    api_key = 'PINECONE_ API_KEY',
    environment="gcp-starter",
)
index_name = "data"

docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)


docsearch = Pinecone.from_existing_index(index_name, embeddings)

retriever = docsearch.as_retriever(search_kwargs={"k": 4})

compressor = CohereRerank(cohere_api_key = 'COHERE_API_KEY')
reranker = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# ------------------------------------------------------------------------------------------------


_template = """
Question:
{question}

Search query:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


Test = """
Question: {question}
---------------------
    {summaries}
---------------------
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(Test)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

# ------------------------------------------------------------------------------------------------

memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, input_key='question', max_token_limit=1000)
question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
answer_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True,prompt=chat_prompt)

chain = ConversationalRetrievalChain(
            retriever=reranker,
            question_generator=question_generator,
            combine_docs_chain=answer_chain,
            verbose=False,
            memory=memory,
            rephrase_question=False
)
# ------------------------------------------------------------------------------------------------


query1 = "ماهو الطقس في الرياض"
query2 = "What is the summary of the discussion of cybersecurity issues at the Riyadh forum?"
query3 = "بماذا تعاهد ولي العهد مع بوتين عن منظمة الاوبك ؟"

# ------------------------------------------------------------------------------------------------

result = chain({"question": query1})
result

result = chain({"question": query2})
result

result = chain({"question": query3})
result

