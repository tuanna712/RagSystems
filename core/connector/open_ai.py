from openai import OpenAI

class MyOAI():
    def __init__(self, api_key:str, chat_model:str="gpt-3.5-turbo-1106"):
        self.chat_model = chat_model
        self.client = OpenAI(api_key=api_key)

    def get_chat(self, prompt:str=None, system:str=None, temp:float=0.0, 
                 stop:str=None, max_tokens:int=2000, stream:bool=False):
        if system==None:
            system = "You are an VPI - an AI assistant developed by Vietnam Petroleum Institue that helps people find information."

        completion = self.client.chat.completions.create(
        model=self.chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
        max_tokens=max_tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stream=stream,
        )
        return completion.choices[0].message.content

    def get_embedding(self, text:str):
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding