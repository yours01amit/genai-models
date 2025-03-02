from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4',temperature=0)

result = model.invoke("Suggest me 5 indial names")

print(result.content)
