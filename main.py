from dotenv import load_dotenv
import pprint

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    res = app.invoke(input={"question": "What is short term memory in agents?"})
    # pprint.pprint(res)
    print('----------------------')
    print('Question: ', res['question'])
    print('Source - Web' if res['web_search'] else 'Source - RAG')
    print(res['generation'])
    print('------------------------')
    res = app.invoke(input={"question": "What is super memory in agents?"})
    # pprint.pprint(res)
    print('----------------------')
    print('Question: ', res['question'])
    print('Source - Web' if res['web_search'] else 'Source - RAG')
    print(res['generation'])