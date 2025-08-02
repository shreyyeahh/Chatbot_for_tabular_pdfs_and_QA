# PDF Table Extractor & Question Answering App
➡️https://tabularpdfqnachatbot.streamlit.app/⬅️

**This is an interactive web application built with Streamlit that allows users to upload PDF documents, automatically extracts tabular data from them, and answers questions based on the extracted information.**

**What It Does**
The primary goal of this application is to bridge the gap between complex, tabular data locked inside PDF files and human-readable insights.

**PDF Table Extraction:** Users can upload any PDF document containing tables. The application uses the docling library to intelligently parse the document and extract all tables into structured DataFrames.

**Dynamic Q&A:** Once the tables are extracted, users can ask natural language questions about the data.

**Intelligent Answering:** The app uses a TF-IDF vectorizer to find the most relevant table(s) to the user's query and then leverages the gpt-4o-mini model to generate a precise and context-aware answer.

**Why I Built It**
This project was built to create a seamless and framework-independent solution for a common data extraction problem. Many valuable datasets are trapped in PDFs, and manually extracting them is tedious and error-prone. By combining powerful open-source libraries like docling and scikit-learn with a state-of-the-art language model, this tool automates the entire process, from raw PDF to actionable answers. It serves as a practical example of building a complete data application from the ground up.

**How to Use the Live App**
Visit the Live App: Click the Live Demo Link at the top of this page.

**Upload a PDF:** Use the sidebar to upload a PDF file that contains one or more tables.

**Wait for Processing:** The app will process the file, which may take a moment. A success message will appear when it's done.

**View Extracted Tables:** The main area will display all the tables that were successfully extracted from the document. You can expand each table to view its contents.

**Ask a Question:** Type your question about the data in the text box and press Enter.

**Get Your Answer:** The application will find the relevant information and generate a direct answer for you.

**Running the App Locally**
If you want to run this application on your own machine, follow these steps:

Prerequisites: Make sure you have Python 3.8+ and Git installed.

# Clone the Repository:
Open your terminal and clone this repository to your local machine.

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Set up a Virtual Environment: It's highly recommended to use a virtual environment.

**Create a virtual environment**
python -m venv venv

**Activate it (Windows)**
.\venv\Scripts\activate

**Activate it (macOS/Linux)**
source venv/bin/activate

**Install Dependencies:** Install all the required packages.

pip install -r requirements.txt

**Provide Your API Key:**

Create a folder named .streamlit in the main project directory.

Inside this new folder, create a file named secrets.toml.

Add your OpenAI API key to this file as shown below:

**.streamlit/secrets.toml**
OPENAI_API_KEY = "your-openai-api-key-goes-here"

Run the App: Now, you can run the Streamlit application.

streamlit run app.py

Technologies Used
Backend: Python

Web Framework: Streamlit

PDF Parsing & Table Extraction: docling

Data Manipulation: pandas

Text Vectorization & Search: scikit-learn (TfidfVectorizer)

AI-Powered Answering: openai (gpt-4o-mini)

**IMPORTANT: API Key Notice**
The live demo of this application uses my personal OpenAI API key to power the question-answering feature. The API is not free and operates on a credit-based system.

If the application stops providing answers to questions, it is likely that the API credits have been exhausted.

In this case, the table extraction will still function correctly, but the Q&A feature will be disabled until the credits are replenished. If you run the app locally, you must provide your own API key as described in the steps above.
