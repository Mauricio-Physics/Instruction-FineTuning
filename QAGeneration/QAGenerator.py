from QAGeneration.QuestionsGenerator import QuestionsGenerator
from QAGeneration.QuestionsEvaluator import QuestionsEvaluator
import os
from dotenv import load_dotenv

class QAGenerator:

    def __init__(self, data_path, output_path, questions_topics='uso e manutenzione', input_pages=2,
                 last_checkpoint_questions=None, last_checkpoint_evaluation=None, last_checkpoint_answers=None):
        '''
        data_path: path to the parsed document (must contain the parsing both in JSON and Markdown format)
        output_path: path to the output file
        questions_topics: the topics about which we want to generate questions
        input_pages: the number of pages we want to generate questions for
        last_checkpoint_questions: path of the last checkpoint for questions generation
        last_checkpoint_evaluation: path of the last checkpoint for questions evaluation
        last_checkpoint_answers: path of the last checkpoint for answers generation
        '''
        self.data_path = data_path
        self.output_path = output_path
        os.system(f'rm -r {output_path}/questions/final/')
        os.system(f'rm -r {output_path}/questions_evaluation/final/')
        os.system(f'mkdir {output_path}/questions')
        os.system(f'mkdir {output_path}/questions_evaluation')
        self.questions_topics = questions_topics
        self.input_pages = input_pages
        load_dotenv()
        self.questions_output_path = output_path+'/questions/'
        self.evaluation_output_path = output_path+'/questions_evaluation/'
        self.last_checkpoint_questions = self.questions_output_path + last_checkpoint_questions
        self.skip_question_generation = False if last_checkpoint_evaluation is None else True
        self.last_checkpoint_evaluation = output_path + last_checkpoint_evaluation if (last_checkpoint_questions is not
                                            None and last_checkpoint_evaluation is not None ) else None
        self.skip_evaluation_generation = False if last_checkpoint_answers is None else True


    def run(self):
        if self.skip_question_generation is False:
            print('-----Run questions generation-----')
            self.run_questions_generation()
        if self.skip_evaluation_generation is False:
            print('-----Run questions evaluation-----')
            self.run_questions_evaluation()
        if self.skip_evaluation_generation is False:
            print('-----Run questions evaluation-----')
            self.run_answers_generation()


    def run_questions_generation(self):
        questions_generator = QuestionsGenerator(input_path=self.data_path, output_path=self.questions_output_path,
                                                 last_checkpoint=self.last_checkpoint_questions)
        questions_generator.generate_questions(input_pages=self.input_pages, topics=self.questions_topics)

    def run_questions_evaluation(self):
        evaluate_questions = QuestionsEvaluator(input_path=self.questions_output_path+'final',
                                                output_path=self.evaluation_output_path)
        evaluate_questions.evaluate()

    def run_answers_generation(self):
        evaluate_questions = QuestionsEvaluator(input_path=self.questions_output_path+'final',
                                                output_path=self.evaluation_output_path)
        evaluate_questions.evaluate()

