the following libraries are needed

- langchain
- dotenv
- flask
- flask-restful
- flasgger
- huggingface-hub
- flask-swagger-ui
- transformers
- torch
- chromadb
- flask-cors


use /api/client/upload end point for file upload
use /api/client/LoadAndRetrieve for query answering

remaining tasks:
- make the lama agent work with the sql ( prombt to call the best one for the job is needed)
- further tests for the sql agent
- save csv into sql to use the sql agent
- make chromadb save into tables not only one table
