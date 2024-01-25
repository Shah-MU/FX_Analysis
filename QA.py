import time
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import LocalAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
import openai
import json
from datetime import datetime
import os

def qa_agent():
    with st.chat_message("Assistant"):
        st.write("Hello feel free to ask me anything about this project!")



    def download_session_state():
        session_state_json = json.dumps(st.session_state.messages, indent=2)
        session_state_bytes = session_state_json.encode("utf-8")

        st.download_button(
            label="Save Conversation (JSON)",
            data=session_state_bytes,
            file_name=f"{datetime.today().strftime('%Y-%m-%d')}.json",
            key="download_session_state",
        )

    def upload_session_state():
        uploaded_file = st.file_uploader("Upload Conversation (JSON)", type="json")

        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8")
            st.session_state.messages = json.loads(content)
            st.sidebar.error('''Select (×) to unmount JSON to continue using the application''')

        if uploaded_file is not None:
            content = uploaded_file.read()
            st.session_state.messages = json.loads(content)

    # Initialize the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    tab1, tab2= st.sidebar.tabs(['Note', 'Save Conversation'])

    with tab1:
        st.markdown('''This is a chat bot has been fine-tuned on this projects wiki using an embedding model.
        Both the embedding model and the LLM are hosted locally, as
        such performance issues may occur due to traffic and question complexity, please allow 1-2 minutes
        for a generated response

        \nIf you wish to interact with a lighter standalone version of this chatbot which has not been fine-tuned click on the link [here!](https://shah-mu-chatbot.streamlit.app/)
        
        \nConversations can be saved and loaded in JSON format using the alternate tab
        ''')

    with tab2:
        upload_session_state()
        download_session_state()

    # Set up the Langchain LLM
    openai.api_type = "open_ai"
    openai.api_base = 'http://144.172.137.100:1234/v1'
    openai.api_key = "NULL"

    # Upload the PDF file
    text_file_path = 'TA_DB.txt'  # Replace 'your_text_file.txt' with the actual path to your text file

    if text_file_path is None or not os.path.exists(text_file_path):
        st.markdown('#')
        st.error('''Please enter a valid path to a text file to continue''')
    else:
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Split the text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings using LocalAIEmbeddings
        embeddings = LocalAIEmbeddings(
            openai_api_base="http://144.172.137.100:8080", model="text-embedding-ada-002", openai_api_key="NULL"
        )

        # Build the knowledge base using FAISS
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Display only user and assistant messages to the end user
        for idx, message in enumerate(st.session_state.messages):
            if message["role"] in ["user", "assistant"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # Add a button to delete the last message
                    if st.button(f"Delete Message", key=f"delete_{message['role']}_{idx}"):
                        st.session_state.messages.pop(idx)
                        st.rerun()

        # React to user input
        if user_question := st.chat_input("Enter To Start Chatting!"):
            st.chat_message("user").markdown(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

            # Perform similarity search and run the QA chain
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI(base_url="http://144.172.137.100:1234/v1", streaming=True, openai_api_key="NULL")
                chain = load_qa_chain(llm, chain_type="stuff")
                with st.spinner("Thinking..."):
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)






            # Stream the assistant's response with a delay
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                assistant_response = response

                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

