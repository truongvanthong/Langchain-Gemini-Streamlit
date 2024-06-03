# Gemini PDF Chatbot
Tiếng Việt: Gemini PDF Chatbot là một ứng dụng dựa trên Streamlit cho phép người dùng trò chuyện với một mô hình AI trò chuyện được đào tạo trên tài liệu PDF. Chatbot trích xuất thông tin từ các tệp PDF được tải lên và trả lời các câu hỏi của người dùng dựa trên ngữ cảnh được cung cấp.

## Các tính năng

- **PDF Upload:** Người dùng có thể tải lên các tệp PDF để trích xuất thông tin.
- **Text Extraction:** Trích xuất văn bản từ tệp PDF và tạo các vectơ nhúng.
- **Conversational AI:** Mô hình AI trả lời các câu hỏi dựa trên ngữ cảnh từ tài liệu PDF.
- **Chat Interface:** Giao diện trò chuyện cho phép người dùng tương tác với mô hình AI.

## Hướng dẫn lấy API key từ Google
<!-- https://aistudio.google.com/app/apikey -->
1. Truy cập vào trang [Google AI Platform](https://aistudio.google.com/app/apikey).
2. Nhấp vào "Create API Key" để tạo một API key mới.
3. Sao chép API key và lưu trữ nó một cách an toàn.

## Cài đặt và sử dụng với Docker

Nếu bạn đã cài đặt docker, bạn có thể chạy ứng dụng bằng lệnh sau:

- Bây giờ bạn lấy API key từ Google và đặt nó trong file `.env`.
   ```.env
   GOOGLE_API_KEY=your_api_key_here
   ```

```bash
docker compose up --build
```
- Mở trình duyệt và truy cập vào địa chỉ `http://localhost:8501`.

## Cài đặt và sử dụng với Python (Local)

   **Lưu ý:** Dự án này yêu cầu Python 3.10 hoặc cao hơn.

1. **Clone the Repository:**

   ```bash
   https://github.com/truongvanthong/Langchain-Gemini-Streamlit.git
   ```

2. **Cài đặt các thư viện:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google API Key:**
   - Lấy API key từ Google và đặt nó trong file `.env`

   ```bash
   GOOGLE_API_KEY='your_api_key_here'
   ```

4. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

5. **Mở trình duyệt và truy cập vào địa chỉ `http://localhost:8501`.**

## Hướng dẫn sử dụng

1. **Chat Interface:**
   - Giao diện chính cho phép người dùng trò chuyện với AI.

2. **Upload PDFs:**
   - Sử dụng thanh bên để tải lên các tệp PDF.
   - Nhấp vào "Submit & Process" để trích xuất văn bản và tạo các vectơ nhúng.

## Cấu trúc dự án

- `app.py`: File chính chứa mã nguồn cho ứng dụng Streamlit.
- `.env`: file chứa các biến môi trường. API key của Google được lưu ở đây.
- `requirements.txt`: Danh sách các thư viện cần thiết cho dự án.
- `README.md`: File hướng dẫn cài đặt và sử dụng.

## Yêu cầu

- streamlit
- google-generativeai
- python-dotenv
- langchain
- PyPDF2
- chromadb
- faiss-cpu
- langchain_google_genai
- langchain_community
- langdetect

## Tác giả

- [Google Gemini](https://ai.google.com/): Mô hình AI trò chuyện được đào tạo trên tài liệu PDF.
- [Streamlit](https://streamlit.io/): Thư viện Python cho việc xây dựng ứng dụng web dữ liệu.
