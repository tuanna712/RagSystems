import os, json
from core.connector.open_ai import MyOAI
from core.connector.qdrantdb import MyQdrant
from dotenv import load_dotenv
load_dotenv()

QDRANT_DB_PATH='database'
COLLECTION_NAME='tndksh'
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

#Check and Delete file "database/.lock"
if os.path.exists(QDRANT_DB_PATH + '/.lock'):
    os.remove(QDRANT_DB_PATH + '/.lock')

OAIClient = MyOAI(api_key=OPENAI_API_KEY)
QDClient = MyQdrant(local_location=QDRANT_DB_PATH)

PROMPT = """
References information is below.\n---------------------\n{context_str}\n---------------------\n
Using both the references information and also using your own knowledge, answer the question: 
{query_str}\n
If the context isn't helpful, you must answer that you dont know.\n
Always answer in Vietnamese.
"""

def retrieve_references(query:str):
    search_res = QDClient.search_data(
        collection_name=COLLECTION_NAME,
        query_vector=OAIClient.get_embedding(query),
        top_k=3,
    )
    references = ""
    references_list = []
    for i, item in enumerate(search_res):
        node = json.loads(item.payload['_node_content'])
        node_content = node['text'].replace('\n', ' ')
        references += f"Reference {i+1}:\n" + node_content + "\n\n"
        references_list.append(node_content)
    return references, references_list

## This is an example of how to use the chatbot
# references =retrieve_references(query)
# answer = OAIClient.get_chat(prompt=PROMPT.format(context_str=references, query_str=query))

#================================================================================================
# Load json file from data/dataset_utf8.json
import json, random

with open('data/dataset_utf8.json') as f:
    data = json.load(f)
    
question_list = []
answer_list = []
contexts_list = []
ground_truths_list = []
ground_contexts = []

for i in range(len(data)):
    #Create random number in [1,2,3]
    question_num = f"q{random.randint(1, 3)}"
    question_list.append(data[f"doc_{i}"]['questions'][question_num])
    ground_contexts.append(data[f"doc_{i}"]['context'])

for i in range(len(question_list)):
    references, references_list =retrieve_references(question_list[i])

    answer = OAIClient.get_chat(prompt=PROMPT.format(context_str=references, query_str=question_list[i]))
    ground_truth = OAIClient.get_chat(prompt=PROMPT.format(context_str=ground_contexts[i], query_str=question_list[i]))

    answer_list.append(answer)
    contexts_list.append(references_list)
    ground_truths_list.append(ground_truth)

    print(f"Question {i+1}: {question_list[i]}")
    print(f"GroundTruth: {ground_truth}")
    print(f"Answer: {answer}")
    print('-----------------------------------')

data_dict = {'question': question_list, 'answer': answer_list, 'context': contexts_list, 'ground_truth': ground_truths_list}
# Save the data_dict to a json file
with open('data/dataset_w_ans.json', 'w') as f:
    json.dump(data_dict, f)

#================================================================================================
from datasets import Dataset 
import json
# Load data from 'data/dataset_w_ans.json' to json
with open('data/dataset_w_ans.json') as f:
    loaded_data = json.load(f)

# Load the data_dict to a dataset
generated_dataset = {
            'question': loaded_data['question'],
            'answer': loaded_data['answer'],
            'contexts' : loaded_data['context'],
            'ground_truth': loaded_data['ground_truth'],
            }
dataset = Dataset.from_dict(generated_dataset)
df_dataset = dataset.to_pandas()
df_dataset.head()
#================================================================================================
import os
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", api_key=OPENAI_API_KEY, temperature=0.0, max_tokens=4000)

eval_df = evaluate(dataset, metrics=[context_recall], llm=llm).to_pandas()
con_preci_eval_df = evaluate(dataset, metrics=[context_precision], llm=llm).to_pandas()
faithf_eval_df = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm).to_pandas()

#Concatenate "context_precision" column of con_preci_eval_df to eval_df
_eval_df = pd.concat([eval_df, con_preci_eval_df['context_precision']], axis=1)
_eval_df = pd.concat([_eval_df, faithf_eval_df['faithfulness']], axis=1)
_eval_df = pd.concat([_eval_df, faithf_eval_df['answer_relevancy']], axis=1)
# Drop "ground_truths" column
_eval_df.drop(columns=['ground_truths'], inplace=True)
_eval_df.head()
# Save the _eval_df to a excel file
_eval_df.to_excel('data/evaluation_result.xlsx', index=False)
#================================================================================================
import matplotlib.pyplot as plt
def plot_score(df, label, color):
    plt.figure(figsize=(10, 6))
    plt.plot(df, color, label=label)
    plt.xlabel('Question')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

# Create new df from columns ['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy']
new_df = _eval_df[['context_recall', 'context_precision', 'faithfulness', 'answer_relevancy']]

# Sort asending by 'context_recall'
new_df = new_df.sort_values(by='context_recall').reset_index(drop=True)
plot_score(new_df['context_recall'], 'context_recall', 'b')

# Sort asending by 'context_precision'
new_df = new_df.sort_values(by='context_precision').reset_index(drop=True)
plot_score(new_df['context_precision'], 'context_precision', 'y')

# Sort asending by 'faithfulness'
new_df = new_df.sort_values(by='faithfulness').reset_index(drop=True)
plot_score(new_df['faithfulness'], 'faithfulness', 'r')

# Sort asending by 'answer_relevancy'
new_df = new_df.sort_values(by='answer_relevancy').reset_index(drop=True)
plot_score(new_df['answer_relevancy'], 'answer_relevancy', 'g')

