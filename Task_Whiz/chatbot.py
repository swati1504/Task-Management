from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

class ChatBot:
    def __init__(self, csv_file_path, api_key):
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = api_key

        # Load the CSV file
        loader = CSVLoader(file_path=csv_file_path)

        # Create a document search index from the CSV data
        index_creator = VectorstoreIndexCreator()
        docsearch = index_creator.from_loaders([loader])

        # Create a RetrievalQA chain
        self.chain = RetrievalQA.from_chain_type(
            llm=OpenAI(), 
            chain_type="stuff", 
            retriever=docsearch.vectorstore.as_retriever(), 
            input_key="question"
        )

    def chat(self, query):
        # Process the query and get a response
        response = self.chain({"question": query})
        return response['result']

