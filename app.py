import os
import re
import time
import json
import csv
import logging
import streamlit as st
import PyPDF2
import requests
import dotenv
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import pinecone
 
# Load environment variables
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
 
# Check if API keys are available
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env file!")
   
if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not found in .env file!")
 
# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
 
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
 
# Global configuration
CONFIG = {
    "output_dir": "scraped_content",
    "max_depth": 3,
    "timeout": 15,
    "concurrent_requests": 5,
    "sleep_between_requests": 1.5,
    "model_name": "gemini-2.0-flash"
}
 
# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)
 
# Initialize session state
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'doc_id' not in st.session_state:
    st.session_state.doc_id = ""
if 'source_type' not in st.session_state:
    st.session_state.source_type = ""  # Track if the data is from PDF or website
 
# Load embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
 
embed_model = get_embedding_model()
 
# Initialize Pinecone client
@st.cache_resource
def init_pinecone():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index_name = "newindex"
    # Check if index exists, create if it doesn't
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            spec=pinecone.PodSpec(
                environment="us-east-1",
                metric="cosine",
                dimension=384  # Set the correct dimension for your embeddings
            )
        )
    
    index = pc.Index(index_name)
    return index
 
index = init_pinecone()
 
# Headers for web requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}
 
# Common Functions
 
def clean_text(text):
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove non-printable characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    return text
 
def store_embeddings(text, doc_id, source_type):
    """Store text embeddings in Pinecone with metadata."""
    try:
        # Split text into sentences or chunks
        sentences = text.split(". ")
 
        if not sentences:
            st.warning("No text to embed.")
            return
 
        # Create embeddings
        with st.spinner("Creating embeddings..."):
            embeddings = embed_model.encode(sentences).tolist()
 
        # Create vectors for Pinecone, including metadata with document ID
        vectors = [(f"{doc_id}_{i}", embeddings[i], {
            "text": sentences[i],
            "source": doc_id,  # Add document ID as metadata
            "source_type": source_type})
                  for i in range(len(sentences))]
 
        # Upload to Pinecone in batches
        batch_size = 100
        total_vectors = len(vectors)
 
        # Initialize the progress bar
        progress_bar = st.progress(0)
 
        for i in range(0, total_vectors, batch_size):
            end_idx = min(i + batch_size, total_vectors)
            batch = vectors[i:end_idx]
            index.upsert(batch)
           
            # Update progress
            progress_bar.progress((end_idx) / total_vectors)
 
        st.success(f"Successfully stored {len(vectors)} embeddings in Pinecone!")
        st.session_state.data_processed = True
        return True
    except Exception as e:
        st.error(f"Error storing embeddings in Pinecone: {e}")
        logger.error(f"Error storing embeddings: {e}")
        return False
 
def retrieve_relevant_text(query):
    """Retrieve relevant text from Pinecone based on query."""
    try:
        query_embedding = embed_model.encode([query]).tolist()[0]
        results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
       
        if results['matches']:
            contexts = []
            sources = set()
            source_types = set()
           
            for match in results['matches']:
                contexts.append(match['metadata']['text'])
                if 'source' in match['metadata']:
                    sources.add(match['metadata']['source'])
                if 'source_type' in match['metadata']:
                    source_types.add(match['metadata']['source_type'])
           
            relevant_text = "\n".join(contexts)
           
            # Make PDF source clickable
            source_info = ""
            for source in sources:
                if source.startswith("pdf_"):  # Check if it's a PDF
                    source_info += f"[PDF: {source}]({CONFIG['output_dir']}/{source}.pdf)\n"
                else:
                    source_info += f"Source: {source}\n"
           
            source_type_info = f"Source Types: {', '.join(source_types)}" if source_types else ""
           
            return relevant_text, source_info, source_type_info
        return "No relevant text found.", "", ""
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        logger.error(f"Error querying Pinecone: {e}")
        return f"Error querying Pinecone: {e}", "", ""
 
def send_to_gemini(query, retrieved_text):
    """Send query and text to Gemini for response generation."""
    try:
        model = genai.GenerativeModel(CONFIG["model_name"])
 
        prompt = f"""
        Given the following query and relevant context, generate a comprehensive and accurate response.
       
        Query: {query}
       
        Context:
        {retrieved_text}
       
        Please provide a detailed answer based solely on the information in the context.
        If the context doesn't contain relevant information to answer the query,
        state that clearly rather than making up information.
        """
       
        response = model.generate_content(prompt)
        return response.text if response else "No response generated."
    except Exception as e:
        st.error(f"Error generating content with Gemini: {e}")
        logger.error(f"Error generating content with Gemini: {e}")
        return f"Error generating content with Gemini: {e}"
 
# PDF Functions
 
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    try:
        with st.spinner("Extracting text from PDF..."):
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            return clean_text(text)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        logger.error(f"Error extracting text from PDF: {e}")
        return ""
 
# Web Scraping Functions
 
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def make_request(url):
    """Make a request with retry capability."""
    try:
        with st.spinner(f"Requesting {url}..."):
            response = requests.get(url, headers=HEADERS, timeout=CONFIG["timeout"])
            response.raise_for_status()
            return response
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request failed for {url}: {e}")
        st.warning(f"Request failed for {url}: {e}")
        raise
 
def extract_text_from_url(url):
    """Extract readable text content from a URL."""
    try:
        logger.info(f"Fetching content from: {url}")
        response = make_request(url)
        soup = BeautifulSoup(response.text, "html.parser")
       
        # Remove non-content elements
        for element in soup(["script", "style", "header", "footer", "nav", "aside", "iframe", "noscript"]):
            element.extract()
           
        # Get text content
        text = soup.get_text(separator=' ')
        return clean_text(text)
    except Exception as e:
        logger.error(f"Failed to extract text from {url}: {e}")
        return None
 
 
def get_all_links(base_url, max_depth=CONFIG["max_depth"], visited=None, current_depth=0):
    """Recursively fetch all internal links up to a specified depth."""
    if visited is None:
        visited = set()
       
    if current_depth >= max_depth:
        return visited
       
    if base_url in visited:
        return visited
       
    visited.add(base_url)
   
    try:
        logger.info(f"Getting links from: {base_url} (depth {current_depth})")
        response = make_request(base_url)
        soup = BeautifulSoup(response.text, "html.parser")
       
        base_domain = urlparse(base_url).netloc
       
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(base_url, href)
            parsed_url = urlparse(full_url)
           
            # Skip external links, PDFs, and other non-HTML resources
            skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', '.docx', '.xls', '.xlsx']
            if (parsed_url.netloc != base_domain or
                any(full_url.lower().endswith(ext) for ext in skip_extensions) or
                '#' in full_url or '?' in full_url or 'mailto:' in full_url):
                continue
               
            if full_url not in visited:
                visited.add(full_url)
               
        # Get next level links (recursive call) but limit concurrent operations
        if current_depth < max_depth - 1:
            new_links = list(visited - {base_url})
            for new_link in new_links[:5]:  # Limit to 5 links per level to avoid excessive crawling
                if new_link not in visited:
                    get_all_links(new_link, max_depth, visited, current_depth + 1)
                    time.sleep(CONFIG["sleep_between_requests"])
                   
        return visited
    except Exception as e:
        logger.error(f"Error getting links from {base_url}: {e}")
        return visited
 
def filter_relevant_links(links, query):
    """Use Gemini AI to filter the most relevant links based on our query."""
    if not links:
        logger.warning("No links to filter")
        return []
       
    logger.info(f"Filtering {len(links)} links using Gemini AI")
   
    prompt = f"""
    Task: Filter the most relevant website links based on this query.
   
    Query: {query}
   
    Website links:
    {links}
   
    Instructions:
    1. Analyze each link and evaluate its potential to contain information related to the query.
    2. Return ONLY a Python list of the most relevant links (maximum 5 links).
    3. Prioritize links containing terms related to the query.
    4. Format your response as a valid Python list: ["url1", "url2", "url3"]
    """
   
    try:
        model = genai.GenerativeModel(CONFIG["model_name"])
        response = model.generate_content(prompt)
       
        if not response or not response.text:
            logger.error("Empty response from Gemini API")
            return []
           
        raw_text = response.text.strip()
       
        # Extract list from response
        list_match = re.search(r'\[(.*?)\]', raw_text, re.DOTALL)
        if list_match:
            # Extract and parse the list
            list_str = list_match.group(0)
            try:
                relevant_links = eval(list_str)
                if isinstance(relevant_links, list) and all(isinstance(item, str) for item in relevant_links):
                    logger.info(f"Successfully filtered to {len(relevant_links)} relevant links")
                    return relevant_links
            except:
                pass
               
        # If regex or parsing fails, try cleaning the text
        cleaned_text = raw_text.strip()
        try:
            relevant_links = eval(cleaned_text)
            if isinstance(relevant_links, list) and all(isinstance(item, str) for item in relevant_links):
                logger.info(f"Successfully filtered to {len(relevant_links)} relevant links")
                return relevant_links
        except:
            logger.warning("Couldn't parse Gemini's response as a list")
            return []
    except Exception as e:
        logger.error(f"Error in filtering links: {e}")
        return []
 
 
def crawl_website(url, query):
    """Crawl website based on query and extract text from relevant pages."""
    with st.spinner("Analyzing website structure..."):
        # Get all links from the website (limited depth)
        all_links = get_all_links(url, max_depth=2)
        st.info(f"Found {len(all_links)} pages on this website")
       
        if not all_links:
            st.warning(f"No links found for {url}")
            # Just extract from the main URL if no links found
            text = extract_text_from_url(url)
            return text if text else ""
       
        # Filter links to find the most relevant ones based on query
        relevant_links = filter_relevant_links(list(all_links), query)
       
        if not relevant_links:
            st.warning("No relevant pages identified, using main page")
            relevant_links = [url]
        else:
            st.success(f"Found {len(relevant_links)} relevant pages")
   
    # Extract text from relevant pages
    full_text = ""
    for link in relevant_links:
        page_text = extract_text_from_url(link)
        if page_text:
            full_text += page_text
        time.sleep(CONFIG["sleep_between_requests"])
   
    return full_text if full_text else ""
 
# Main Application Code (Streamlit UI)
 
st.title("PDF and Website Content Processing App")
 
tabs = st.tabs(["Upload PDF", "Enter Website URL"])
 
with tabs[0]:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
 
    if uploaded_file:
        # Reset session state when uploading new file
        st.session_state.extracted_text = ""
        st.session_state.doc_id = ""
        st.session_state.source_type = "pdf"
       
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(uploaded_file)
 
        if extracted_text:
            st.session_state.extracted_text = extracted_text
            st.session_state.doc_id = f"pdf_{uploaded_file.name}"
            st.success(f"Successfully extracted {len(extracted_text)} characters from PDF")
           
            # Store embeddings
            if st.button("Process PDF", key="process_pdf"):
                store_embeddings(extracted_text, st.session_state.doc_id, "pdf")
 
 
with tabs[1]:
    url = st.text_input("Enter website URL", "https://")
 
    if url:
        # Reset session state when entering new URL
        st.session_state.extracted_text = ""
        st.session_state.doc_id = ""
        st.session_state.source_type = "website"
 
        if st.button("Crawl Website", key="crawl_website"):
            # Crawl the website and extract content
            extracted_text = crawl_website(url, "company information")
 
            if extracted_text:
                st.session_state.extracted_text = extracted_text
                st.session_state.doc_id = f"web_{urlparse(url).netloc}"
                st.success(f"Successfully extracted content from website")
 
                # Store embeddings
                store_embeddings(extracted_text, st.session_state.doc_id, "website")
 
# Querying Section
 
if st.session_state.data_processed:
    st.markdown("---")
    st.subheader("Ask Questions About the Content")
    query = st.text_input("Enter your query:")
 
    if query:
        if st.button("Generate Answer", key="generate"):
            with st.spinner("Retrieving relevant information..."):
                retrieved_text, source_info, source_type_info = retrieve_relevant_text(query)
 
            if retrieved_text and retrieved_text != "No relevant text found.":
                with st.spinner("Generating AI response..."):
                    response = send_to_gemini(query, retrieved_text)
 
                # Display results
                st.markdown("### AI Response:")
                st.markdown(response)
 
                # Show sources
                if source_info:
                    st.markdown("#### Sources:")
                    st.markdown(source_info)
 
                if source_type_info:
                    st.markdown(f"#### Source Type: {source_type_info}")
 
                # Optionally show context
                with st.expander("Show retrieved context"):
                    st.markdown(retrieved_text)
            else:
                st.warning("No relevant information found for your query. Try a different question.")
 
 
 