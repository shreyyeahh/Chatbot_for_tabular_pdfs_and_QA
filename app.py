import streamlit as st
import pandas as pd
import os
import textwrap
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# --- Helper Functions (from your Colab notebook) ---

def extract_data_with_docling(input_data_path):
    """
    Extracts tables from a PDF file using the Docling library.
    """
    try:
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        doc_converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = doc_converter.convert(input_data_path)

        all_tables_data = []
        if not result.document.tables:
            st.warning("No tables were found in the uploaded PDF.")
            return pd.DataFrame()

        for table_ix, table in enumerate(result.document.tables):
            table_df: pd.DataFrame = table.export_to_dataframe()

            # --- FIX for Duplicate Columns ---
            # Check if the dataframe has duplicate column names
            if table_df.columns.has_duplicates:
                new_columns = []
                counts = {}
                for col in table_df.columns:
                    # Ensure column name is a string
                    col_name = str(col)
                    if col_name in counts:
                        counts[col_name] += 1
                        # Append a suffix to make it unique
                        new_columns.append(f"{col_name}_{counts[col_name]}")
                    else:
                        counts[col_name] = 0
                        new_columns.append(col_name)
                table_df.columns = new_columns
            # --- END FIX ---

            # Clean up the dataframe to ensure it's just text
            table_df = table_df.fillna('').astype(str)
            table_content_string = ' '.join([' '.join(row) for row in table_df.values])
            all_tables_data.append({
                "table_number": table_ix + 1,
                "table_content": table_content_string,
                "dataframe": table_df # Store the dataframe for display
            })
        summary_df = pd.DataFrame(all_tables_data)
        return summary_df
    except Exception as e:
        st.error(f"An error occurred during table extraction with Docling: {e}")
        return pd.DataFrame()


def create_keyword_index(df: pd.DataFrame):
    """
    Creates and returns a TF-IDF vectorizer and matrix for keyword search.
    """
    if 'table_content' not in df.columns or df.empty:
        return None, None
    corpus = df['table_content'].tolist()
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def retrieve_top_chunks(query, vectorizer, original_df, tfidf_matrix, top_n=1):
    """
    Retrieves the top N most similar chunks for a given query.
    """
    if vectorizer is None or tfidf_matrix is None:
        return pd.DataFrame()
    query_vector = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[::-1][:top_n]
    results = original_df.iloc[top_indices].copy() # Use .copy() to avoid SettingWithCopyWarning
    results['similarity_score'] = cosine_sim[top_indices]
    return results

def generate_final_answer(query: str, retrieved_chunks_df: pd.DataFrame) -> str:
    """
    Generates a final answer using GPT-4o-mini based on retrieved chunks.
    """
    if 'OPENAI_API_KEY' not in st.secrets:
        st.error("OpenAI API key is not set. Please add it to your Streamlit secrets.")
        return "API key not configured."

    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except openai.AuthenticationError:
        return "OpenAI API key is invalid."

    if retrieved_chunks_df.empty:
        return "Could not retrieve relevant information from the document to answer the query."

    context = "\n---\n".join(retrieved_chunks_df['table_content'])
    answer_prompt = f"""
    Using ONLY the context provided below, give a direct and comprehensive answer to the user's question.
    If the context does not contain the information, state that the answer is not available in the provided document tables.

    Context:
    ---
    {context}
    ---
    User's Question: {query}

    Final Answer:
    """
    try:
        answer_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful question-answering assistant that strictly uses the provided context from document tables to answer questions."},
                {"role": "user", "content": answer_prompt}
            ],
            temperature=0.1
        )
        final_answer = answer_response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate a final answer. Error: {e}"

    return final_answer

# --- Streamlit App UI ---

st.set_page_config(page_title="PDF Table Extractor & QA", layout="wide")

st.title("📄 PDF Table Extractor and Question Answering")
st.markdown("Upload a PDF with tables, and I'll extract the data and answer your questions about it.")

# --- State Management ---
# We use session_state to store data across reruns
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = pd.DataFrame()
if 'tfidf_vectorizer' not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None
if 'processed_file_name' not in st.session_state:
    st.session_state.processed_file_name = ""


# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("1. Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # To prevent reprocessing the same file on every interaction
        if uploaded_file.name != st.session_state.processed_file_name:
            with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
                # Save the uploaded file temporarily to pass its path to Docling
                temp_dir = "temp_files"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 1. Extract data
                st.session_state.extracted_data = extract_data_with_docling(temp_file_path)

                # 2. Create TF-IDF index
                if not st.session_state.extracted_data.empty:
                    st.session_state.tfidf_vectorizer, st.session_state.tfidf_matrix = create_keyword_index(st.session_state.extracted_data)
                    st.session_state.processed_file_name = uploaded_file.name
                    st.success(f"Successfully processed '{uploaded_file.name}'!")
                else:
                    st.warning("Could not extract any tables or content to index.")

                # Clean up the temporary file
                os.remove(temp_file_path)


# --- Main Content Area ---
if not st.session_state.extracted_data.empty:
    st.header("2. Extracted Tables")
    st.info(f"Found {len(st.session_state.extracted_data)} table(s) in the document.")

    for index, row in st.session_state.extracted_data.iterrows():
        with st.expander(f"Table {row['table_number']}"):
            st.dataframe(row['dataframe'])

    st.header("3. Ask a Question")
    user_query = st.text_input("Enter your question about the tables:", key="query_input")

    if user_query:
        with st.spinner("Searching for the answer..."):
            # Retrieve relevant chunks
            top_results = retrieve_top_chunks(
                user_query,
                st.session_state.tfidf_vectorizer,
                st.session_state.extracted_data,
                st.session_state.tfidf_matrix
            )

            # Generate final answer
            final_answer = generate_final_answer(user_query, top_results)

            st.subheader("📝 Answer")
            st.markdown(textwrap.fill(final_answer, width=80))

            with st.expander("Show Retrieved Context"):
                if not top_results.empty:
                    st.write("The answer was generated based on the following extracted table content:")
                    st.dataframe(top_results[['table_number', 'similarity_score']])
                    st.text(top_results.iloc[0]['table_content'])
                else:
                    st.warning("No relevant context was found for this query.")
else:
    st.info("Please upload a PDF file to get started.")