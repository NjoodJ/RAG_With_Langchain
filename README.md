# RAG with Langchain
This Python script utilizes various libraries to create a conversational AI model that can retrieve and summarize information from specified web pages. The model leverages OpenAI's APIs, Cohere for embeddings, and Pinecone for vector storage, allowing it to answer questions about specific topics, such as the weather in Riyadh or recent news events.

<p align="center">
  <img width="600" height="200" src=https://github.com/user-attachments/assets/8f28fa9b-fa55-421a-9f40-942454f6b8ab>
</p>

## Dependencies
Ensure you have the following Python libraries installed:

* tiktoken
* openai
* tqdm
* pinecone-client
* langchain

## You can install them using pip:

```bash
pip install tiktoken openai tqdm pinecone-client langchain
```
## Configuration
Before running the script, you'll need to set up your API keys for the following services:

* OpenAI: For language model access.
* Cohere: For text embeddings.
* Pinecone: For vector storage.

Replace the placeholders in the script with your actual API keys:

```bash
cohere_api_key = 'YOUR_COHERE_API_KEY'
openai_api_key = 'YOUR_OPENAI_API_KEY'
pinecone_api_key = 'YOUR_PINECONE_API_KEY'
```
## Usage
* Input URLs: The script is initialized with a list of URLs to scrape information from. You can modify the urls list to include any pages you want to retrieve data from.
* Loading Data: The WebBaseLoader is used to load data from the specified URLs. This data is then processed and split into manageable chunks.
* Embeddings and Vector Store: Cohere embeddings are created for the documents, which are then stored in Pinecone.
* Retrieval Chain: A conversational retrieval chain is established, allowing the model to answer questions based on the retrieved documents.
* Querying: You can query the model by modifying the query1, query2, and query3 variables with your desired questions.

## Example Queries
The script includes example queries that demonstrate how to interact with the model:
```python
query1 = "ماهو الطقس في الرياض"  # Arabic: What is the weather in Riyadh?
query2 = "What is the summary of the discussion of cybersecurity issues at the Riyadh forum?"
query3 = "بماذا تعاهد ولي العهد مع بوتين عن منظمة الاوبك ؟"  # Arabic: What did the Crown Prince promise Putin regarding OPEC?
```
## Output
The results of the queries are printed out in the console. Each query will return a response based on the retrieved context from the web documents.
<p align="center">
  <img width="1000" height="110" src=https://github.com/user-attachments/assets/381994ef-3dc3-4cb9-be79-2f3e503d8b12>
</p>

## Notes
Ensure that the APIs are accessible and that your network allows outbound connections to the respective services.
The script is designed to provide conversational capabilities, so feel free to expand the queries or modify the templates to fit your needs.












