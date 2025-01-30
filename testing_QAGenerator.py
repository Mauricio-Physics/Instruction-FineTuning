import argparse
from QAGeneration.QAGenerator import QAGenerator
from dotenv import load_dotenv


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path of the parsed file (both JSON and Markdown)')
    parser.add_argument('--output_path', type=str, help='Folder where you want to save generated data')
    parser.add_argument('--questions_topics', type=str, help='Topics to generate questions')
    parser.add_argument('--input_pages', type=str, help='Number of pages to use to generate questions')
    return parser.parse_args()


def main():
    args = parse_command_line_arguments()
    qa_gen = QAGenerator(data_path=args.data_path, output_path=args.output_path, questions_topics=args.questions_topics,
                         input_pages=args.input_pages)
    qa_gen.run()


if __name__ == "__main__":
    load_dotenv()
    main()



