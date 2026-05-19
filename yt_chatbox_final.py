import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


st.title("Youtube chatbox")
video_id = st.text_input("Enter the Id of the video")
if st.button("Submit"):
    st.session_state.video_loaded = True
    
if st.session_state.get("video_loaded"):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=['en'])
        transcript = " ".join(chunck.text for chunck in transcript_list)
    except TranscriptsDisabled:
        print("No caption available for the video")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    vector_store = FAISS.from_documents(chunks, embeddings)
    retrieval = vector_store.as_retriever(search_type='similarity', search_kwargs={"k":4})
    prompt = PromptTemplate(
        template="""
            You are a helpful AI assistant
            Answer only from provided transcript context.
            If transcript context is insufficient, just say you dont know.
            {context}
            question = {question}
    """,
    input_variables=['context', 'question']
    )
    api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    llm = HuggingFaceEndpoint(
        model="Qwen/Qwen3-8B",
        task = "text_generation"
    )
    model = ChatHuggingFace(llm=llm)
    question = st.text_input("Ask anything about the video")
    if question:
        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text
        parallel_chain = RunnableParallel({
            'context':retrieval | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        parser = StrOutputParser()
        mainchain = parallel_chain | prompt | model | parser

        st.write(mainchain.invoke(question))
