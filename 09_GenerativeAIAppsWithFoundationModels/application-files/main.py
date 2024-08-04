# From Cell 2 with small modifications
import os
import streamlit as st
import json
import boto3
import logging

from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import Chroma, FAISS
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# From Cell 5
sm_llm_embeddings = HuggingFaceEmbeddings()

# From Cell 4 and 6
class ContentHandler(LLMContentHandler):
    
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]

def query_endpoint_with_json_payload(encoded_json, endpoint_name, content_type="application/json"):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=encoded_json
    )
    return response

def parse_response_model(query_response):
    model_predictions = json.loads(query_response["Body"].read())
    return [gen["generated_text"] for gen in model_predictions]


# The following replaces cells 8 and 9
# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    elif extension == '.csv':
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(file)
    else:
        print('Document format is not supported!')
        return None

    document = loader.load()
    return document


# From Cell 11
def split_text(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
    texts = text_splitter.split_documents(document)
    return texts


# Cell 11
def create_embeddings(texts):
    docsearch = FAISS.from_documents(texts, sm_llm_embeddings)
    return docsearch

# Not in notebook but needed for streamlite application
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


# Application build from notebook - see individual parts
def ask_and_get_answer(question, documents):
    from langchain_community.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
    from langchain.chains.question_answering import load_qa_chain
    
    # From Cell 13
    docs = documents.similarity_search(question, k=3)
    
    # From Cell 14
    prompt_template = """You are an AI assistant for answering questions.
    Refrane from providing any information that is not in the provide context.
    If there is not an answer in the provided context respond with "I don't know."
    {context}
    
    Question: {question}

    Answer:"""

    parameters ={
        "max_new_tokens": 100,
        "num_return_sequences": 1,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": False,
        "return_full_text": False,
        "temperature": 0.2
    }

    # From Cell 4
    _MODEL_CONFIG_ = {
    
     "jumpstart-dft-hf-llm-falcon-7b-instruct-bf16" : {
        "aws_region": "us-east-1",
        "endpoint_name": "jumpstart-dft-hf-llm-falcon-7b-instruct-bf16",
        "parse_function": parse_response_model,
        "prompt": prompt_template,
        },
    
    }
    
    # From Cell 6
    content_handler = ContentHandler()

    sm_llm_falcon_instruct = SagemakerEndpoint(
        endpoint_name=_MODEL_CONFIG_["jumpstart-dft-hf-llm-falcon-7b-instruct-bf16"]["endpoint_name"],
        region_name=_MODEL_CONFIG_["jumpstart-dft-hf-llm-falcon-7b-instruct-bf16"]["aws_region"],
        model_kwargs=parameters,
        content_handler=content_handler,
    )
    
    # From Cell 10
    sm_llm_falcon_instruct.model_kwargs = {
        "max_new_tokens": 50,
        "num_return_sequences": 1,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": False,
        "return_full_text": True,
        "temperature": 0.1,
        }
    
    
    # From Cell 14 and 15
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(llm=sm_llm_falcon_instruct, prompt=PROMPT)

    answer = chain({"input_documents": docs, "question": question}, return_only_outputs=True)["output_text"]
    
    return answer



# Code required for the Streamlite app
if __name__ == "__main__":
    import os
    st.subheader('Retrieval Augmented Generation (RAG)')
    with st.sidebar:
        # file uploader widget
        uploaded_file = st.file_uploader('Upload context file:', type=['pdf', 'docx', 'txt', 'csv'])
        
        # add data button widget
        add_data = st.button('Process Context File', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ... Please Wait'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                document = load_document(file_name)
                texts = split_text(document)

                # creating the embeddings and returning FAISS vector store.
                vector_store = create_embeddings(texts)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File processing completed successfully!  You can now ask questions.')
                
    
    # user's question text input widget
    question = st.text_input('Ask a question about the content of your file:')
    
    if question: # if the user entered a question and hit enter
        question = f"{question}"
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            # st.write(f'k: {k}')
            response = ask_and_get_answer(question, vector_store)
            answer = response.partition("Answer:")[2]

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer, height=400)

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {question} \nA: {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history')