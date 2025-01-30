import os
import re
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from lib.utilities import (prepare_document, setup_chat_model, extract_json_files, extract_machine_name,
                           split_document_pages, remove_answers, renumber_questions, save_questions,
                           save_question_final)
from datasets import Dataset, load_from_disk


class QuestionsGenerator:
    '''
    This class is demanded to generate a dataset of question given industrial manuals and documentation.
    '''
    def __init__(self, input_path, last_checkpoint=None, checkpoints=True, output_path='data/questions/'):
        self.model_name = os.environ['MODEL_NAME']
        self.llm = setup_chat_model(self.model_name)
        self.input_path = input_path
        self.json_list = extract_json_files(input_path)
        if last_checkpoint is not None:
            self.json_list = self.get_docs_from_checkpoint(last_checkpoint)
        else:
            self.json_list = set(self.json_list)
        self.checkpoints = checkpoints
        self.huggingface_repo = None if os.environ.get('HUGGINGFACE_REPO') is None else (
                                os.environ.get('HUGGINGFACE_REPO'))
        self.output_path = output_path

    def get_docs_from_checkpoint(self, last_checkpoint_path):
        '''
        :param last_checkpoint_path:path of the last checkpoint used to restore the generatiob
        :return: set of document for which no question has been generated
        '''
        documents = set(load_from_disk(last_checkpoint_path)['document'])
        documents = [i.replace('.pdf', '.json') for i in documents]
        return set(self.json_list) - set(documents)

    def generate_questions(self, input_pages, topics):
        '''
        :param input_pages: number of consecutive pages used to generate a batch of questions
        :param topics: topics used to generate the questions
        :return: nothing (create and save huggingface dataset locally and, eventually, push on huggingface HUB
        '''
        for doc in tqdm(self.json_list):
            try:
                questions, pages, machine_code, docs = self.extract_questions_simple(self.input_path + doc, topics,
                                                                input_pages)
                save_questions(questions, self.input_path+doc, first_page=pages, machine_code=machine_code,
                               chunks=docs, input_pages=input_pages, output_path=self.output_path)
            except Exception as e:
                print(e)
        save_question_final(huggingface_repo=self.huggingface_repo, output_path=self.output_path)

    def extract_questions_simple(self, document_path, topics, input_pages):
        """
        Input:
        - document: a string containing the document's content.
        - topics: a string (ITALIAN) containing the topics to generate questions about.
            The text should be formulated to be included in the system sentence described below.
        """
        questions_simple_filtered = []
        docs = []
        pages = []
        document = prepare_document(document_path)
        n_pages = re.findall(r"page_\d+", document)
        if len(n_pages) < 3:
            print(
                "Document {} is too short, we assume it does not contain useful information for question generation".format(
                document_path))
            return None
        machine_name, machine_code = extract_machine_name(document_path)
        if machine_name is None:
            return None
        split_document = split_document_pages(document, input_pages=input_pages)
        count = 0
        for doc in split_document:
            input_pages = int(input_pages)
            start_page = count * input_pages + 1
            system = (f"Devi generare domande relative al macchinario, parendo dalla documentazione fornita input."
                        f"Le domande devono riguardare {topics} (non includere domande sul copyright)."
                        f"Le domande devono utilizzare il nome specifico del macchinario: {machine_name}."
                        f"Le domande devono essere in italiano. Le domande devono essere generiche e non troppo specifiche\n"
                        f"È fondamentale che le informazioni necessarie per rispondere alle domande siano contenute nel "
                        f"documento fornito in input. Se la risposta non è contenuta nel documento oppure la risposta "
                      f"potrebbe essere parziale, allora non generare alcune domanda. Meglio essere più coservativi."
                        )
            user = (f"Genera domande secondo le istruzioni precedentemente descritte ma non le risposte.\n"
                    f"Se non ci sono informazioni sufficienti per generare coppie di domande e risposte, ritorna una "
                    f"lista vuota.\n"
                    f"È fondamentale che le informazioni necessarie per rispondere alle domande siano contenute nel "
                    f"documento fornito in input."
                    f"Le domande devono utilizzare il nome specifico del macchinario: {machine_name}.\n"
                    f"Se il documento contiene un indice non generare domande."
                    f"Il testo sulla base del quale devi generate le domande è:\n{doc}\n"
                    )
            prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
            chain = prompt | self.llm
            try:
                doc_questions_simple = chain.invoke({})
                doc_questions_simple = doc_questions_simple.content
                doc_questions_simple = doc_questions_simple.split('\n')
                if doc_questions_simple is not None:
                    doc_questions_simple_filtered = remove_answers(doc_questions_simple)
                for i in doc_questions_simple_filtered:
                    pages.append(start_page)
                    docs.append(doc)
                questions_simple_filtered += doc_questions_simple_filtered
            except Exception as e:
                print(e)
            count += 1
        questions_simple_filtered = renumber_questions(questions_simple_filtered)
        return questions_simple_filtered, pages, machine_code, docs







