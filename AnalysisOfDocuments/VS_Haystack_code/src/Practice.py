from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
import json
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents
import pandas as pd

#init api
app = FastAPI()

@app.post('/QuerySDG')
async def QuerySDG(SDG, file: UploadFile = File(...)):
    data_file = file.file.read()
    data = json.loads(data_file.decode('utf-8'))
    
    # In-Memory Document Store
    document_store = InMemoryDocumentStore()

    retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    model_format="sentence_transformers")

    total_doc_count = 50000
    batch_size = 256

    counter = 0
    docs = []
    for text in data['elements']:
    # create haystack document object with text content and doc metadata
     if 'Text' in text:
      doc = Document(
        content=text["Text"],
        meta={
            "PageNo": text["Page"]
        } )
      docs.append(doc)
      counter += 1

     if counter % batch_size == 0:
        # writing docs everytime `batch_size` docs are reached
        embeds = retriever.embed_documents(docs)
        for i, doc in enumerate(docs):
            doc.embedding = embeds[i]
        document_store.write_documents(docs)
        docs.clear()
     if counter == total_doc_count:
        break

    search_pipe = DocumentSearchPipeline(retriever)

    return search_pipe.run(
        query=SDG,
        params={"Retriever": {"top_k": 2}}
        )


@app.post('/UploadSDGs')
async def UploadSDGs(SDG: UploadFile = File(...), Companyfile: UploadFile = File(...)):

    SDGfile = SDG.file.read()
    SDGs = json.loads(SDGfile.decode('utf-8'))

    Companyfile = Companyfile.file.read()
    CompanyAgendas = json.loads(Companyfile.decode('utf-8'))
    
    # In-Memory Document Store
    document_store = InMemoryDocumentStore()

    retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    model_format="sentence_transformers")

    total_doc_count = 50000
    batch_size = 256

    counter = 0
    docs = []
    for text in CompanyAgendas['elements']:
    # create haystack document object with text content and doc metadata
     if 'Text' in text:
      doc = Document(
        content=text["Text"],
        meta={
            "PageNo": text["Page"]
        } )
      docs.append(doc)
      counter += 1

     if counter % batch_size == 0:
        # writing docs everytime `batch_size` docs are reached
        embeds = retriever.embed_documents(docs)
        for i, doc in enumerate(docs):
            doc.embedding = embeds[i]
        document_store.write_documents(docs)
        docs.clear()
     if counter == total_doc_count:
        break
    
    # results = []
    # for sdg in SDGs['elements']:
    # # create haystack document object with text content and doc metadata
    #  if 'Text' in sdg:
    #   sdg_text = sdg["Text"]
    # #   print(sdg_text)
    #   search_pipe = DocumentSearchPipeline(retriever)
    #   result = search_pipe.run(
    #       query=sdg_text,params={"Retriever": {"top_k": 1}})
    #   results.append(result)
    # return(results)

    results = []
    SDG_text_list = []
    Similarity_list = []
    Agenda_list = []

    for sdg in SDGs['elements']:
    # create haystack document object with text content and doc metadata
     if 'Text' in sdg:
      sdg_text = sdg["Text"]
    #   print(sdg_text)
      search_pipe = DocumentSearchPipeline(retriever)
      result = search_pipe.run(
          query=sdg_text,params={"Retriever": {"top_k": 1}})
      results.append(result)

      SDG_list = SDG_text_list.append(sdg_text)
      Similarityscore = result.get('score')

      Similarity_list.append(Similarityscore)
      Agenda = result.get('content')
      Agenda_list.append(Agenda)

    # if FileResponse(media_type = 'application/json'): 
    #     return(results)
    # elif FileResponse(media_type = 'application/json'):
    #     headers = {'Content-Disposition': 'attachment; filename="Book.xlsx"'}
    #     return FileResponse('excel_file_path', headers=headers)
    
    dict = {'SDG':SDG_text_list,
        'Similarity': Similarity_list,
        'CompanyAgenda':Agenda_list}
        
    df = pd.DataFrame(dict)
    
    return StreamingResponse(
        iter([df.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=output.csv"})

    





