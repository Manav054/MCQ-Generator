import os
import PyPDF2
import json
import traceback

def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception("Error reading the file!")
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        raise Exception("Unsupported file format... Only pdf and txt file supported!!")
    
def get_table_data(quiz):
    try:
        quiz = quiz.strip().splitlines()[1:-1]  # Removes delimiters and empty lines
        quiz = "".join(quiz)
        quiz = json.loads(quiz)
        quiz_table_data = []
        for key, value in quiz.items():
            mcq = value["mcq"]
            options = " || ".join(
                [
                    f"{option}: {option_value}"
                    for option, option_value in value["options"].items()
                ]
            )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
        return quiz_table_data
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
