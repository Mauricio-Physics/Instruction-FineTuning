from lib.utilities import setup_chat_model
import os
from datasets import load_from_disk, Dataset
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm


class AnswersEvaluation:
    def __init__(self, input_path, output_path):
        load_dotenv()
        self.model_name = os.environ['MODEL_NAME']
        self.llm = setup_chat_model(self.model_name)
        self.input_path = input_path
        self.output_path = output_path

    @staticmethod
    def create_prompt(ds):
        to_return = []
        for i in ds:
            user = (f"### Contesto\n {i['context']}\n\n"
                    f"{i['chunks_rag']}"
                    f"### Domanda\n {i['question']}\n\n")
            #to_return.append(ChatPromptTemplate.from_messages([("system", system), ("human", user)]))
            to_return.append(user)
        return to_return

    def run(self):
        ds = load_from_disk(self.input_path)[:20]
        ds = Dataset.from_dict(ds)
        users = self.create_prompt(ds)
        questions = []
        answers = []
        answers_reference = []
        systems = []
        for i in tqdm(range(len(users))):
            questions.append(ds[i]['question'])
            answers_reference.append(ds[i]['answer'])
            systems.append(ds[i]['system'])
            prompt = ChatPromptTemplate.from_messages([("system", ds[i]['system']), ("human", users[i])])
            chain = prompt | self.llm
            answer = chain.invoke({})
            answers.append(answer.content)

        output_dict = {
            'system': systems,
            'user': users,
            'question': questions,
            'answer_reference': answers_reference,
            'answers': answers,
        }
        ds = Dataset.from_dict(output_dict)
        ds.save_to_disk(self.output_path)


