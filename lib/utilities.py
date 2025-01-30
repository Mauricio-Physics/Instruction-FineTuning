import os
from langchain.chat_models import AzureChatOpenAI
import pandas as pd
import json
import re
from datasets import Dataset, load_from_disk, concatenate_datasets
import numpy as np
from lib.CustomElasticSearch import CustomElasticSearch
from elasticsearch import Elasticsearch


def split_document_pages(document, input_pages=1):
    """
    Given a document, it splits it into pages (according to page_i delimiters) and then joins them in groups of input_pages
    """
    pages = re.split(r'page_\d+', document)
    pages = [page.strip() for page in pages if page.strip()]

    pages_join = [''.join(pages[i:i+input_pages]) for i in range(2, len(pages), input_pages)]
    return pages_join


def setup_chat_model(model_name='gpt-35', max_tokens=200):
    '''
    :param model_name: name of the LLM which we want to use to generate question and/or answers
    :return: Callable object used to invoke LLM
    '''

    if model_name == 'gpt-35':
        print(f"Using GPT-3.5 Turbo.")
        '''
        llm = AzureChatOpenAI(
                azure_endpoint=os.environ['AZURE_ENDPOINT'],
                openai_api_version="2023-03-15-preview",
                openai_api_key=os.environ['AZURE_API_KEY'],
                deployment_name=os.environ['AZURE_MODEL'],
                model=os.environ['AZURE_MODEL'],  # Name of the deployment for identification
            )
        '''
    elif model_name == 'vllm':
        print(f"Using Mistral 7B")
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(api_key='EMPTY',
                    base_url=os.environ['VLLM_ENDPOINT'],
                     model=os.environ['VLLM_MODEL'],
                    temperature=0,
                    max_tokens=max_tokens)

    else:
        raise ValueError('Model not recognized')
    return llm


def extract_json_files(input_path):
    files_list = os.listdir(input_path)
    json_list = []
    for i in files_list:
        if i.endswith('json') and i.replace('json', 'pdf'):
            json_list.append(i)
    return json_list


def extract_machine_name(document_name,):
    """
    Input: Document
    Output: The name of the machine described in the document if found, otherwise None.
    Given a document, the function uses the model to extract the name of the machine from the document.
    To reduce the number of tokens usage, only the first page of the document is passed.
    """
    machines = pd.read_csv('info_files.csv')
    print(document_name.replace('json', 'pdf').replace('parsed_azure/', ''))
    machines = machines[machines['filename'] == document_name.replace('json', 'pdf').replace('parsed_azure/', '')]
    machines_name = machines['machine_name']
    machines_code = machines['machine_code']
    return machines_name.tolist()[0], machines_code.tolist()[0]


def prepare_document(document_path):
    """
    Takes as input a document path and returns a string with the document's content.
    Some preprocessing is applied to remove curly braces from the document as they can cause issues with the LLM model.
    """
    with open(document_path, 'r') as file:
        document = json.load(file)

    document = [list(document.keys())[i] + ' ' + list(document.values())[i] for i in range(len(document))]
    document = ''.join(document)
    document = document.replace('}', '')
    document = document.replace('{', '')
    return document


def remove_answers(questions_list):
    """
    Input:
    - questions_list: A list of questions (and potentially answers which are removed by filtering).
    Output:
    - filtered_questions_list: A list of questions without any answers.
    Filter to remove answers from the list of questions.
    """
    # Remove them if they do not contain a question mark.
    questions_list = [x for x in questions_list if '?' in x]

    # Remove them if they start by Risposta or Response or Answer
    pattern = r"^(Risposta|Response|Answer)"
    questions_list = [x for x in questions_list if not re.match(pattern, x)]

    return questions_list


def renumber_questions(questions):
    """
    Input list of questions and renumber them
    """
    questions_no_number = [re.sub(r'^\d+\.', '', x) for x in questions]
    questions_number = [f'{idx+1}. {x}' for idx, x in enumerate(questions_no_number)]
    return questions_number


def save_questions(input_list, document_path, input_pages, machine_code, first_page, chunks,
                   output_path='data/questions/'):
    '''
    :param input_list: list of questions
    :param document_path: path of the document used to generate the questions
    :param input_pages: number of pages used to generate the questions
    :param machine_code: code of the machine
    :param first_page: first page used to generate eack question
    :param chunks: document used to generate the questions
    :param dataset_name: fina name og the dataset
    :return:
    '''
    if input_list is None:
        print(f"No questions generated.")
        return
    question_or_answer = 'questions'
    input_list = [re.sub(r'^\d+\.', f'{question_or_answer} {idx+1}.', x) for idx, x in enumerate(input_list)]
    filename_without_extension = re.search(r'([^\/]+)(?=\.\w+$)', document_path).group(1)
    document_names = [filename_without_extension + '.pdf' for i in input_list]
    input_pages_list = [input_pages for i in input_list]
    machine_code_list = [machine_code for i in input_list]
    list_checkpoints = os.listdir(output_path)
    list_checkpoints = [int(i.replace('checkpoint_', '')) for i in list_checkpoints]
    list_checkpoints.append(0)
    last_checkpoint = np.max(list_checkpoints)
    if last_checkpoint > 0:
        ds = load_from_disk(output_path + 'checkpoint_{}'.format(last_checkpoint))
    else:
        ds = None
    data = {'question': input_list,
                'first_page': first_page,
                'document': document_names,
                'input_pages': input_pages_list,
                'machine_code': machine_code_list,
                'chunks': chunks
            }
    ds_to_append = Dataset.from_dict(data)
    if ds is not None:
        ds = concatenate_datasets([ds, ds_to_append])
    else:
        ds = ds_to_append
    ds.save_to_disk(output_path + 'checkpoint_{}'.format(last_checkpoint+1))


def save_question_final(huggingface_repo=None, output_path='data/questions/'):
    list_checkpoints = os.listdir(output_path)
    list_checkpoints = [int(i.replace('checkpoint_', '')) for i in list_checkpoints]
    list_checkpoints.append(0)
    last_checkpoint = np.max(list_checkpoints)
    ds = load_from_disk(output_path + 'checkpoint_{}'.format(last_checkpoint))
    ds.save_to_disk(output_path + 'final')
    if huggingface_repo is not None:
        ds.push_to_hub(huggingface_repo)

def create_load_retriever(elasticsearch_url, index_name, recreate: bool = False, embedding_model_name: str = None):
    """
    If recreate is true the index is deleted and recreated.
    Otherwise try to create the retriever and if it already exists load it instead.
    """
    print(f"Embedding model name: {embedding_model_name}")
    if recreate == True:
        retriever = CustomElasticSearch.create(elasticsearch_url=elasticsearch_url, index_name=index_name,
                                                      recreate=True, embedding_model_name=embedding_model_name)
    else:
        try:
            retriever = CustomElasticSearch.create(elasticsearch_url=elasticsearch_url,
                                                          index_name=index_name, recreate=False, embedding_model_name=embedding_model_name)  # Create index
        except Exception as e:
            print("Failed to create index. Error:", e)
            try:
                retriever = CustomElasticSearch(client=Elasticsearch(elasticsearch_url),
                                                       index_name=index_name, embedding_model_name=embedding_model_name)
            except Exception as e:
                print("Failed to create or load the retriever. Error:", e)
                return
    return retriever

def save_answer_dataset(data, name):
    data.save_to_disk(name)
