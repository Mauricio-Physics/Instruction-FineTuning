import os
import re
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from lib.utilities import setup_chat_model
from datasets import Dataset, load_from_disk
import pandas as pd
import numpy as np

class QuestionsEvaluator:
    def __init__(self, input_path, output_path, cot=False):
        self.model_name = os.environ['MODEL_NAME']
        self.input_path = input_path
        self.llm = setup_chat_model(self.model_name)
        self.output_path = output_path
        self.ds = None
        self.cot = False

    def evaluate(self, checkpoint_steps=500):
        self.ds = pd.DataFrame(load_from_disk(self.input_path))
        documents = set(self.ds['document'].tolist())
        self.ds['relevance'] = np.nan * len(self.ds)
        self.ds['global_relevance'] = np.nan * len(self.ds)
        self.ds['coverage'] = np.nan * len(self.ds)
        counter = 0
        for doc in tqdm(documents):
            filtered_doc = self.ds[self.ds['document'] == doc]
            first_pages = set(filtered_doc['first_page'].tolist())
            for page in first_pages:
                filtered_dataset = filtered_doc[filtered_doc['first_page'] == page]
                questions = filtered_dataset['question']
                idxs = filtered_dataset.index.tolist()
                questions_string = ''
                try:
                    for q in questions:
                        questions_string += q + '\n'
                    relevance = self.calculate_relevance(questions=questions_string, document=doc, cot=self.cot)
                    global_relevance = self.calculate_global_relevance(questions=questions, cot=self.cot)
                    coverage = self.calculate_coverage(document=doc, questions=questions_string, cot=self.cot)
                    if len(coverage)==1:
                        coverage = [coverage[0] for i in range(len(questions))]
                    elif len(coverage) != len(questions):
                        raise ValueError(f'Number of questions {len(questions)} does not match number of scores '
                                         f'{len(coverage)}(coverage)')

                    if len(relevance) == 1:
                        relevance = [relevance[0] for i in range(len(questions))]
                    elif len(relevance) != len(questions):
                        raise ValueError(f'Number of questions {len(questions)}does not match number of scores '
                                         f'{len(relevance)}(relevance)')

                    if len(global_relevance) == 1:
                        global_relevance = [global_relevance[0] for i in range(len(questions))]
                    elif len(global_relevance) != len(questions):
                        raise ValueError('Number of questions does not match number of scores (global_relevance)')

                    for i in range(len(questions)):
                            self.ds.loc[idxs[i], 'relevance'] = relevance[i]
                            self.ds.loc[idxs[i], 'global_relevance'] = global_relevance[i]
                            self.ds.loc[idxs[i], 'coverage'] = coverage[i]
                            counter += 1
                            if counter + 1 % checkpoint_steps == 0:
                                ds_checkpoint = Dataset.from_pandas(self.ds)
                                ds_checkpoint.save_to_disk(self.output_path + 'checkpoint_{n}'.format(n=counter))
                except Exception as e:
                    print(e)
                    for i in idxs:
                       self.ds = self.ds.drop(i)
        ds_final = Dataset.from_pandas(self.ds)
        ds_final.save_to_disk(self.output_path + 'final')

    def calculate_relevance(self, questions, document, cot=True):
        """
        Input:
        - document: the part of document from which the questions were generated.
        - questions: A list of questions to be evaluated.
        - cot: A boolean indicating whether to use the chain of thought prompt.
        Output:
        - relevance_scores: A list of scores from 1 to 5, where 5 is the most relevant and 1 is the least relevant.

        The goal of this function is to evaluate each question with a relevance score (wrt to the document) from 1 to 5,
        where 5 is the most relevant and 1 is the least relevant.
        To do so, a LLM is used. If cot is set to True, an additional chain of thought prompt is passed in the pipeline.
        """
        system = (f" Il tuo ruolo è valutare quanto è plausibile che un addetto macchine possa avere la necessità di "
                  f"conoscere la risposta a una domanda, dato il documento tecnico in input.\n"
                  f"L'operaio cerca la risposta alla domanda su un documento che verrà fornito.\n"
                  f"Assegna un unico punteggio da 1 a 10, dove 10 rappresenta una domanda rilevante e 1 una domanda non "
                  f"pertinente, tenendo conto del contesto fornito dato dal documento."
                  f"Il punteggio deve essere inserito tra parentesi quadre."
                  f"L'output deve essere come segue:"
                  f"Domanda 1: [score] [comment]"
                  f"Domanda 2: [score] [comment]"
                  f"Domanda 3: [score] [comment]"
                  f"..."
                  f"Dove 'score' è il punteggio assegnato a ciascuna domanda e 'comment' è l'eventuale commento"
                  f"che vuoi aggiungere al punteggio numerico. Tutto il testo generato deve essere in Italiano"
                  f"Presta particolare attenzione nel seguire il formato sopra descritto.")

        if cot:
            user_cot = (f"Quali caratteristiche deve avere una domanda rispetto al contesto del documento per "
                        f"avere un punteggio di 10?")
            prompt_cot = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot)])
            chain_cot = prompt_cot | self.llm
            risposta_cot = chain_cot.invoke({}).content

        relevance_scores = [None] * len(questions)
        user = (f"Valuta se è plausibile che un addetto macchine cerchi le seguenti domande sul documento.\n\n"
                f"Documento: {document}\n\n"
                f"Domande: {questions}\n\n")
        if cot:
            prompt = ChatPromptTemplate.from_messages(
                [("system", system), ('user', user_cot), ('assistant', risposta_cot), ('user', user)])
        else:
            prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
        chain = prompt | self.llm
        chain_output = chain.invoke({}).content
        chain_output = chain_output.replace('(', '[').replace(')', ']')
        relevance_scores = [int(x) for x in re.findall(r'\[(\d+)\]', chain_output)]
        return relevance_scores

    def calculate_global_relevance(self, questions,  cot=True):
        """
            Input:
            - questions: A list of questions to be evaluated.
            - cot: A boolean indicating whether to use the chain of thought prompt.
            Output:
            - global_relevance_scores: A list of scores from 1 to 5, where 5 is the most relevant and 1 is the least
                relevant.

            The goal of this function is to evaluate each question with a score from 1 to 5, where 5 is the most
                relevant and 1 is the least relevant.
            To do so, a LLM is used. If cot is set to True, an additional chain of thought prompt is passed in the
                pipeline.
            """

        system = (f" Il tuo ruolo è valutare quanto è plausibile che un addetto macchine possa avere la necessità di "
                  f"conoscere la risposta a una domanda.\n"
                  f"Nello speficifo la domande viene posta da un operatore in un azienda manifatturiera ad un assistente"
                  f"viruale, capace di rispondere a domande specifiche su uso, manutenzione e specifiche tecniche dei "
                  f"vari macchinari. Alcune domande possono riguardare anche le misure di sicurezza e preventive\n"
                  f"Assegna un unico punteggio da 1 a 10, dove 10 rappresenta una domanda rilevante e 1 una domanda non "
                  f"pertinente. Non riportare il numero della domanda nel testo generato.")
        # Replace to avoid the model printing also the question number and hence the score being extracted as the
        # question number
        questions = [re.sub(r'questions \d+\.', '', x).strip() for x in questions]
        if cot:
            user_cot = f"Quali caratteristiche deve avere una domanda per avere un punteggio di 10?"
            prompt_cot = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot)])
            chain_cot = prompt_cot | self.llm
            risposta_cot = chain_cot.invoke({}).content

        global_relevance_scores = [None] * len(questions)

        for idx, q in enumerate(questions):
            user = f"La domanda da valutare è:\n{q}\n"
            if cot:
                prompt = ChatPromptTemplate.from_messages(
                    [("system", system), ('user', user_cot), ('assistant', risposta_cot), ('user', user)])
            else:
                prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])

            chain = prompt | self.llm
            chain_content = chain.invoke({}).content

            score = re.search(r'\d+', chain_content)
            global_relevance_scores[idx] = score.group(0)
        global_relevance_scores = [int(i) for i in global_relevance_scores]
        return global_relevance_scores

    def calculate_coverage(self, questions, document, cot=True):
        """
        Given a document and a list of questions, it evaluates the coverage of the questions, i.e. whether their answers
        can be found in the associated documents.
        The evaluation is done using a chat model, that assigns a score from 1 to 5, with 1 meaning the answer can not
        be formulated based on the document, while 5 means that a nice answer can be extracted from the document.
        If cot is True, an additional chain of thought prompt is passed in the pipeline.
        """
        system = (f"Il tuo ruolo è valutare se la risposta a una domanda può essere formulata con le informazioni "
                  f"fornite nel contesto.\n"
                  f"Assegna un unico punteggio da 1 a 10 a ogni domanda, dove 10 rappresenta una domanda la cui risposta è "
                  f"contenuta nel contesto e 1 una domanda alla quale non si può rispondere con le informazioni del "
                  f"documento.\n"
                  f"Il punteggio deve essere inserito tra parentesi quadre."
                  f"L'output deve essere come segue:"
                  f"Domanda 1: [score] [comment]"
                  f"Domanda 2: [score] [comment]"
                  f"Domanda 3: [score] [comment]"
                  f"..."
                  f"Dove 'score' è il punteggio assegnato a ciascuna domanda e 'comment' è l'eventuale commento"
                  f"che vuoi aggiungere al punteggio numerico. Tutto il testo generato deve essere in Italiano"
                  f"Presta particolare attenzione nel seguire il formato sopra descritto.")

        if cot:
            user_cot = f"Quali caratteristiche deve avere una domanda per avere un punteggio di 10?"
            prompt_cot = ChatPromptTemplate.from_messages([("system", system), ('user', user_cot)])
            chain_cot = prompt_cot | self.llm
            risposta_cot = chain_cot.invoke({}).content
        user = (f"Documento: {document}\n\n"
                f"Domande: {questions}\n\n")
        if cot:
            prompt = ChatPromptTemplate.from_messages(
                [("system", system), ('user', user_cot), ('assistant', risposta_cot), ('user', user)])
        else:
            prompt = ChatPromptTemplate.from_messages([("system", system), ('user', user)])
        chain = prompt | self.llm
        chain_output = chain.invoke({}).content
        chain_output = chain_output.replace('(', '[').replace(')', ']')
        coverage_scores = [int(x) for x in re.findall(r'\[(\d+)\]', chain_output)]
        return coverage_scores


    def filter_dataset(self, min_score):
        pass

