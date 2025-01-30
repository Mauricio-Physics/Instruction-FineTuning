from lib.utilities import setup_chat_model, create_load_retriever
from datasets import load_from_disk, Dataset
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
import psycopg2
from langchain_text_splitters import TokenTextSplitter
import os, glob
import re
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from lib.utilities import setup_chat_model, create_load_retriever, save_answer_dataset


class AnswersGenerator:
    '''
    Given a dataset composed by question and the document from which they are generated, this class generates answers.
    '''
    def __init__(self, questions_path, last_checkpoint, output_path, data_ingestion=False, document_path='parsed_azure/'):
        load_dotenv()
        self.llm = setup_chat_model(model_name=os.environ['MODEL_NAME'])
        self.questions_path = questions_path
        self.last_checkpoint = last_checkpoint
        self.output_path = output_path
        self.index_name = os.environ['ELASTICSEARCH_INDEX_NAME'] if (
                os.environ['ELASTICSEARCH_INDEX_NAME'] != '') else None
        self.retriever=None
        self.initialize_retriever()
        if data_ingestion:
            data_ingestion = DataIngestion(self.retriever, document_path)
            data_ingestion.run()

    def load_checkpoint(self):
        last_checkpoint_ds = load_from_disk(self.last_checkpoint)
        systems = last_checkpoint_ds['system']
        questions = last_checkpoint_ds['question']
        questions_set = set(questions)
        contexts = last_checkpoint_ds['context']
        answers = last_checkpoint_ds['answer']
        rag_contexts = last_checkpoint_ds['chunks_rag']
        return systems, questions_set, contexts, answers, rag_contexts

    def initialize_retriever(self):
        if self.index_name is not None:
            elasticsearch_url = 'http://{elastic_username}:{elastic_password}@{elastic_host}:{elastic_port}'.format(
                elastic_username=os.environ['ELASTICSEARCH_USERNAME'],
                elastic_password=os.environ['ELASTICSEARCH_PASSWORD'],
                elastic_host=os.environ['ELASTICSEARCH_HOST'],
                elastic_port=os.environ['ELASTICSEARCH_PORT']
            )
            recreate = False if os.environ['RECREATE_INDEX'] == 'False' else True
            self.retriever = create_load_retriever(elasticsearch_url, self.index_name.lower(), recreate=recreate,
                                              embedding_model_name=None)

    def answer_without_rag(self, ds_questions, checkpoint=100):
        systems = []
        contexts = []
        questions = []
        answers = []
        rag_contexts = []
        documents = []
        questions_set = ()
        counter = 0
        if self.last_checkpoint is not None:
            systems, questions_set, contexts, answers, rag_contexts = self.load_checkpoint()
        for i in tqdm(ds_questions):
            try:
                if i['question'] not in questions_set:
                    system = (f"Sei un assistente virtuale e devi rispondere alla domanda considerando solamente le "
                              f"informazioni fornite nel contesto. "
                              f"Le informazioni fornite in input sono in formato Markdown e sono tratte da "
                              f"documentazione tecnica\n"
                              f"La risposta deve contenere il nome del file e la pagina contenente le informazioni "
                              f"necessarie per rispondere alla domanda.\n\n "
                              f"Rispondi solamente in italiano. Non limitarti a fornire l'indicazione della pagina ma "
                              f"genera una risposta completa, con tutte le informazioni necessarie (le risposte non "
                              f"devono essere parziali)\n "
                              f"Presta attenzione a riportare il numero di pagina corretto (le informazioni relative al"
                              f" numero della pagina sono riportate nel footer)")
                    user = (f"###Nome del file\n {i['document']}\n\n ### Contesto\n {i['chunks']}\n\n "
                            f"### Domanda\n {i['question']}\n\n")
                    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
                    chain = prompt | self.llm
                    answer = chain.invoke({})
                    systems.append(system)
                    questions.append(i['question'])
                    contexts.append(i['chunks'])
                    answers.append(answer.content)
                    documents.append(i['document'])
                    counter += 1
                    if self.retriever is not None:
                        retrieved_chunks = self.retriever.retrieve(i['question'], i['document'], 10)
                        all_chunks = [x.page_content for x in retrieved_chunks]
                        rag_context = all_chunks[-1]
                        rag_contexts.append(rag_context)
                    if counter % checkpoint == 0:
                        dataset_name = self.output_path + 'checkpoint_{}'.format(counter)
                        if self.retriever is None:
                            output_dict = {
                                'system': systems,
                                'question': questions,
                                'context': contexts,
                                'answer': answers,
                                'document': documents
                            }
                        else:
                            output_dict = {
                                'system': systems,
                                'question': questions,
                                'context': contexts,
                                'context_rag': rag_contexts,
                                'answer': answers,
                                'document': documents
                            }
                        ds = Dataset.from_dict(output_dict)
                        save_answer_dataset(data=ds, name=dataset_name)
            except Exception as e:
                print(e)
        if self.retriever is None:
            return systems, contexts, questions, answers
        else:
            return systems, contexts, questions, answers, rag_contexts

    def answer_from_context(self):
        question_ds = load_from_disk(self.questions_path)
        if self.index_name is None:
            systems, contexts, questions, answers = self.answer_without_rag(question_ds)
            output_dict = {
                'system': systems,
                'question': questions,
                'context': contexts,
                'answer': answers
            }
        else:
            systems, contexts, questions, answers, rag_contexts = self.answer_without_rag(question_ds)
            output_dict = {
                'system': systems,
                'question': questions,
                'context': contexts,
                'answer': answers,
                'chunks_rag': rag_contexts
            }
        dataset_name = self.output_path + 'final'
        ds = Dataset.from_dict(output_dict)
        save_answer_dataset(data=ds, name=dataset_name)

    def generate(self):
        return self.answer_from_context()


class DataIngestion:
    def __init__(self, retriever, documents_path):
        self.retriever = retriever
        self.conn = psycopg2.connect(database=os.environ['PSQL_DB'],
                            user=os.environ['PSQL_USERNAME'],
                            host=os.environ['PSQL_HOST'],
                            password=os.environ['PSQL_PASSWORD'],
                            port=os.environ['PSQL_PORT'])
        self.chunk_size=os.environ['CHUNK_SIZE']
        self.overlap=os.environ['OVERLAP']
        self.documents_path = documents_path

    def preprocess_data(self):
        cursor = self.conn.cursor()
        files = []
        for file in glob.glob(self.documents_path + "*.md"):
            files.append(file.replace(self.documents_path, ''))
        print(files)
        query_machine_code = 'SELECT d.document_machine FROM documents d where document_filename like \'{doc_name}%\''
        tags_list = []

        for i in files:
            sql_query = query_machine_code.format(doc_name=i[:-3])
            cursor.execute(sql_query)
            results = cursor.fetchall()[0][0]
            tags_list += [results]
        docs = []
        tags = []
        for i in range(len(files)):
            loader = TextLoader(self.documents_path + files[i])
            documents = loader.load()
            text_splitter = TokenTextSplitter(chunk_size=int(self.chunk_size), chunk_overlap=int(self.overlap))
            docs_temp = text_splitter.split_documents(documents)
            for j in docs_temp:
                j.metadata['tag'] = tags_list[i]
                j.metadata['file'] = files[i]
                j.page_content = j.page_content.replace('<figure>\n', '')
                j.page_content = j.page_content.replace('</figure>\n', '')
                j.page_content = re.sub('!\[]\(figures/.*?\)', '', j.page_content, flags=re.DOTALL)
            docs += docs_temp
        return docs
    def run(self):
        docs = self.preprocess_data()
        self.retriever.add_texts(docs)




