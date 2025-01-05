import json
from langchain.schema import Document

def load_and_convert_to_documents(file_paths):
    """Reads files and converts them into LangChain documents."""
    documents = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            try:
                for line in content.splitlines():
                    data = json.loads(line)
                    documents.append(Document(page_content=json.dumps(data)))
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {path}")
    return documents
