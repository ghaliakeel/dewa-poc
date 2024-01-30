import glob
import io
import json
import os

from dotenv import load_dotenv
from langchain import OpenAI, SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from flask import Flask, request,jsonify, render_template
from flask_restful import Resource, Api
from flask_swagger_ui import get_swaggerui_blueprint
from flasgger import Swagger
from langchain.memory import ConversationBufferMemory
from torch import cuda
from constants import CHROMA_SETTINGS, QA_CHAIN_PROMPT
from huggingface_hub import login
from langchain.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
import pandas as pd
from transformers import AutoTokenizer
from langchain.agents import load_tools
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.vectorstores import Chroma
import chromadb
from flask_cors import CORS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
from transformers import StoppingCriteria, StoppingCriteriaList
import ingest
import torch

app = Flask("Ml_img_extract_client")
api = Api(app)
swagger = Swagger(app)
# resources={r"/api/*": {"origins": "http://localhost:5000"}}
CORS(app)
persist_directory = os.environ.get('PERSIST_DIRECTORY')
db_sql = SQLDatabase.from_uri(os.environ.get('SQL_db'))
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'


load_dotenv()
MODEL = os.environ.get('MODEL_ID')
login(os.environ.get('HUGGINGFACEHUB_API_TOKEN'))
tokenizer = AutoTokenizer.from_pretrained(MODEL)


stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


stopping_criteria = StoppingCriteriaList([StopOnTokens()])
pipeline = transformers.pipeline(
    task="text-generation", #task
    model=MODEL,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    repetition_penalty=1.1,  # without this output begins repeating
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # device_map="auto",
    device=0,
    max_length=4048,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline)
embeddings = GPT4AllEmbeddings()


class DocUpload(Resource):
    def post(self):
        """
                Upload an image and extract features.

                ---
                parameters:
                  - name: image
                    in: formData
                    type: file
                    required: true
                    description: The image file to upload.
                  - name: user_data
                    in: formData
                    type: file
                    required: true
                    description: A JSON file with user data, including a 'name' field.

                responses:
                  200:
                    description: File uploaded successfully, it will save the features along with the name provided.
                  400:
                    description: Bad request, check the error message for details.
                """
        # try:
        text,  csv_chunks, message, code = ingest.process_documents(request)

        if code == 200:
            message, code = ingest.store_into_vector(text,csv_chunks)

        if code != 200:
            return message, code
        # except:
        #     return {'error': 'upload the correct format'}, 400
        print(text)
        # encoding, message, code = _extract_features(img=file.read(), user_id=user_id, save=True)
        # if message:
        #     return message, code

        return {'message': 'file ok'}, 200

    def get(self):
        """
                Get information on how to use this endpoint.

                ---
                responses:
                  200:
                    description: Information on using the endpoint.
                """
        return {'use' : 'the follwoing format',
                'file' : 'send a file with the key name file it can be pdf, csv, powerpoint, word',}

#

class LoadAndRetrieve(Resource):

    def post(self):
        file = request.json
        query = file['query']
        args = ingest.parse_arguments()
        chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS,
                    client=chroma_client)

        retriever = db.as_retriever(search_kwargs={"k": 2, "fetch_k": 40}, search_type="mmr")

        callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
        memory = ConversationBufferMemory(memory_key='chat_history')
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
            get_chat_history=lambda h: h,
            callbacks=callbacks)
        FORMAT_INSTRUCTIONS = """Given an input question, first create a syntactically correct MySQL query to run,  
                                then look at the results of the query and return the answer. you can not create new tables, delete tables or update tables.
                                The question: {question}
                                
                                Use the following format:
                                Question: the input question you must answer
                                Thought: you should always think about what to do, what action to take
                                Action: SQl agent
                                Action Input: the input to the action, never add backticks "`" around the action input
                                Observation: the result of the action
                                ... (this Thought/Action/Action Input/Observation can repeat 2 times)
                                Thought: I now know the final answer
                                Final Answer: the final answer to the original input question"""
        toolkit = SQLDatabaseToolkit(db=db_sql, llm=llm)
        sql_agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            combine_docs_chain_kwargs={"prompt": FORMAT_INSTRUCTIONS}
        )
        info_sql_database_tool_description = """Input to this tool is a comma separated list of tables, output is the schema and sample rows for those tables.Be sure that the tables actually exist by calling list_tables_sql_db first! Example Input: table1, table2, table3"""
        Tools = [
        #     Tool(
        #         name='lama QA System',
        #         func=conv_chain.run,
        #         description="this is useful for text data, Input should be a fully formed question.",
        #         return_direct=True,
        # ),

            # Tool(
            #     name='sql agent',
            #     func=sql_agent.run,
            #     description="this is useful for dataset queries or csv or tsv or xls or table, Input should be a fully formed question.",
            #     return_direct=True,
            # ),
            QuerySQLDataBaseTool(db=db_sql),
            InfoSQLDatabaseTool(db=db_sql, description=info_sql_database_tool_description),
            ListSQLDatabaseTool(db=db_sql),
            QuerySQLCheckerTool(db=db_sql, llm=llm),
        ]


        # assign your llm and db



        tools = [

        ]
        agent_instructions = "Try 'SQl agent' tool first, Use the other tools if these don't work."
        agent = initialize_agent(
            Tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            # agent_instructions=agent_instructions,
            max_iterations = 4,
            early_stopping_method = 'generate',
            memory = memory
        )
        result = agent.run(query)
        return {'message': f"{result}"}, 200


api.add_resource(DocUpload, '/api/client/upload')
# api.add_resource(Predict, '/api/client/predict')
api.add_resource(LoadAndRetrieve, '/api/client/LoadAndRetrieve')

SWAGGER_URL = '/docs'
API_URL = '/static/client.json'

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Ml_img_extract_client"}
)

app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

#
# @app.route(API_URL)
# def swagger_spec():
#     return jsonify(api.__schema__)


if __name__ == '__main__':
    DEBUG = True
    app.run()