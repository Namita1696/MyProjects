from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
import json
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents
from sklearn.metrics import precision_recall_fscore_support

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
    document_store = InMemoryDocumentStore(return_embedding=True)

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
    
    results = []
    for sdg in SDGs['elements']:
    # create haystack document object with text content and doc metadata
     if 'Text' in sdg:
      sdg_text = sdg["Text"]
    #   print(sdg_text)
      search_pipe = DocumentSearchPipeline(retriever)
      result = search_pipe.run(
          query=sdg_text,params={"Retriever": {"top_k": 1}})
      results.append(result)
    return(results)

    

from sklearn.metrics import precision_recall_fscore_support

def evaluate_results(results, SDGs):
    # flatten the list of search results
    flattened_results = [result[0] for result in results]

    # extract the true labels of the SDGs
    true_labels = [sdg["Label"] for sdg in SDGs["elements"]]

    # predict the labels of the search results
    predicted_labels = []
    for result in flattened_results:
        if len(result) == 0:
            predicted_labels.append(None)
        else:
            predicted_labels.append(result.meta["Label"])

    # calculate Precision, Recall, and F1-score
    precision, recall, f1_score, support = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted", zero_division=0)

    return precision, recall, f1_score



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
    # # if FileResponse(media_type = 'application/json'): 
    # #     return(results)
    # # elif FileResponse(media_type = 'application/json'):
    # #     headers = {'Content-Disposition': 'attachment; filename="Book.xlsx"'}
    # #     return FileResponse('excel_file_path', headers=headers)
    
    # dict = {'SDG':[sdg_text],
    #     'degree': ["MBA", "BCA", "M.Tech", "MBA"],
    #     'score':[90, 40, 80, 98]}
        
    # df = pd.DataFrame(dict)
    # df = pd.DataFrame(
    #     [["SDG", sdg_text], ["Similarity", 20], ["CompanyAgenda", result]], 
    #     columns=["Ref_text", "Similarity","Ana_text"]
    # )
    # return StreamingResponse(
    #     iter([df.to_csv(index=False)]),
    #     media_type="text/csv",
    #     headers={"Content-Disposition": f"attachment; filename=data.csv"})

    





