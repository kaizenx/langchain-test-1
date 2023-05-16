from llama_index import GPTVectorStoreIndex, download_loader, ServiceContext, LLMPredictor, PromptHelper
from llama_index.node_parser import SimpleNodeParser
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
import os

os.environ["OPENAI_API_KEY"] = "SOME API"


# Set maximum input size
max_input_size = 1024
# Set number of output tokens
num_output = 256
# Set maximum chunk overlap
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

SimpleWebPageReader = download_loader("SimpleWebPageReader")

loader = SimpleWebPageReader()

documents = loader.load_data(urls=["https://www.linkedin.com/in/tcwu78/"])

# parser = SimpleNodeParser()
# nodes = parser.get_nodes_from_documents(documents)
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

# service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512, prompt_helper=prompt_helper)
# NOTE: set a chunk size limit to < 1024 tokens 
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

tools = [
    Tool(
        name="Website Index",
        func=lambda q: query_engine.query(q),
        description=f"asdf.",
    ),
]
print("7")
llm = OpenAI(model_name="text-davinci-003", temperature=0)
print("8")
memory = ConversationBufferMemory(memory_key="chat_history")
print("9")
agent_chain = initialize_agent(
    tools, llm, agent="zero-shot-react-description", memory=memory
)
print("10")
question = "Who is Tzu Chjeh Wu?"
output = agent_chain.run(input=question)
print(question)
print(output)