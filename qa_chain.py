from langchain_community.llms import HuggingFaceHub

def load_qa_chain(docs, device):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    vector_store = Chroma.from_documents(docs, embedding=embeddings)

    # ✅ Use online hosted model
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 512
        },
        huggingfacehub_api_token="" # hf token
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
