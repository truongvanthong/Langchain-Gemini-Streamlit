import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langdetect import detect

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Bạn là một trợ lý ảo thông minh, được thiết kế để cung cấp câu trả lời về các vấn đề dựa trên các tài liệu tham khảo đã cho. Vui lòng làm theo các hướng dẫn sau:
    - Đọc và phân tích tài liệu tham khảo.
    - Sử dụng thông tin từ tài liệu để trả lời câu hỏi đã cho.
    - Nếu không tìm thấy đủ thông tin trong tài liệu, hãy báo rằng "không có thông tin đủ" và gợi ý cho người dùng thêm thông tin vào câu hỏi để rõ ràng hơn.
    - Trả lời với các số liệu, thời gian, hoặc đường dẫn đến nội dung có trong tài liệu nếu có.
    - Trả lời câu hỏi một cách ngắn gọn và tự nhiên nhất.

    Ngữ cảnh:
    {context}

    Câu hỏi:
    {question}

    Trả lời:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",
                                   client=genai,
                                   temperature=0.9)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]
    st.session_state.context = ""


def user_input(user_question):
    detected_language = detect(user_question)
    language = "English" if detected_language == 'en' else "Vietnamese"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists("faiss_index"):
        raise FileNotFoundError(
            "The FAISS index file does not exist. Please process some PDFs first.")
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    # Append current question to the context
    context = st.session_state.context
    if context:
        context += "\n" + user_question
    else:
        context = user_question

    response = chain({"input_documents": docs, "context": context,
                     "question": user_question}, return_only_outputs=True)

    # Update the session state context
    st.session_state.context = context

    return response


def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖")
    if "context" not in st.session_state:
        st.session_state.context = ""
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Tải lên các tài liệu PDF và nhấn nút Submit & Process", accept_multiple_files=True)

        if not pdf_docs:
            st.warning("Vui lòng tải lên các tài liệu PDF")
        else:
            # Kiểm tra xem người dùng đã nhấn nút Submit & Process chưa, nếu nhấn rồi thì sẽ thực hiện các bước sau
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                    st.balloons() # Hiển thị biểu tượng bóng bay   

    st.title("Chat với tài liệu PDF sử dụng Gemini")
    st.write("Hãy tải lên các tài liệu PDF và đặt câu hỏi cho trợ lý ảo")

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    st.sidebar.markdown(
        '<p style="color:#f63366; font-size: 24px">Web By: Trương Văn Thông</p>',
        unsafe_allow_html=True
    )
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
