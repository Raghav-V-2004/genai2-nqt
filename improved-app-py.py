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
import uuid

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
    "model_name": "gemini-2.0-flash",
    "chunk_size": 500,  # Characters per chunk for better context management
    "chunk_overlap": 100,  # Overlap between chunks to maintain context
    "top_k_results": 5  # Fetch top 5 most relevant chunks
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
if 'source_url' not in st.session_state:
    st.session_state.source_url = ""  # Store original URL for website sources
if 'current_source' not in st.session_state:
    st.session_state.current_source = None  # Track current active source
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = ""  # Store PDF file path for reference

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

def chunk_text(text, chunk_size=CONFIG["chunk_size"], overlap=CONFIG["chunk_overlap"]):
    """Split text into overlapping chunks of specified size."""
    chunks = []
    if not text:
        return chunks

    # Simple chunking by character count with overlap
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Find the last period or space before the end to avoid cutting in the middle of sentences
        if end < text_len:
            # Try to find the last period within the last 100 characters of the chunk
            last_period = text.rfind('. ', max(start, end - 100), end)
            if last_period > start:
                end = last_period + 1  # Include the period
            else:
                # If no period found, try with space
                last_space = text.rfind(' ', max(start, end - 50), end)
                if last_space > start:
                    end = last_space

        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end

    return chunks

def reset_session_for_new_source():
    """Reset session state when processing a new source."""
    # Keep track of what's being cleared
    old_source = st.session_state.doc_id

    # Reset session state for new input
    st.session_state.extracted_text = ""
    st.session_state.doc_id = ""
    st.session_state.source_type = ""
    st.session_state.source_url = ""
    st.session_state.data_processed = False
    st.session_state.pdf_path = ""

    # Log the reset
    if old_source:
        logger.info(f"Reset session state from previous source: {old_source}")

    # Optional: we could clear the relevant vectors from Pinecone here
    # but that might not be desirable if you want to keep a knowledge base

def save_pdf_for_reference(uploaded_file):
    """Save uploaded PDF file for future reference."""
    try:
        # Create a unique filename to avoid collisions
        pdf_filename = f"{st.session_state.doc_id}.pdf"
        pdf_path = os.path.join(CONFIG["output_dir"], pdf_filename)

        # Save the file
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.session_state.pdf_path = pdf_path
        logger.info(f"Saved PDF file to {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"Error saving PDF file: {e}")
        st.error(f"Could not save PDF file for reference: {e}")
        return None

def store_embeddings(text, doc_id, source_type, source_url=""):
    """Store text embeddings in Pinecone with metadata."""
    try:
        # Split text into chunks with overlap
        chunks = chunk_text(text)

        if not chunks:
            st.warning("No text to embed.")
            return False

        # Create embeddings
        with st.spinner("Creating embeddings..."):
            embeddings = embed_model.encode(chunks).tolist()

        # Create vectors for Pinecone, including metadata
        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            # Create a unique ID for each chunk
            chunk_id = f"{doc_id}_{i}"

            # Prepare metadata with source information
            metadata = {
                "text": chunk,
                "source": doc_id,
                "source_type": source_type,
                "chunk_index": i
            }

            # Add URL for website sources
            if source_type == "website" and source_url:
                metadata["url"] = source_url

            # Add vectors to our batch
            vectors.append((chunk_id, emb, metadata))

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

        st.success(f"Successfully stored {len(vectors)} text chunks in Pinecone!")
        st.session_state.data_processed = True
        st.session_state.current_source = doc_id
        return True
    except Exception as e:
        st.error(f"Error storing embeddings in Pinecone: {e}")
        logger.error(f"Error storing embeddings: {e}")
        return False

def retrieve_relevant_text(query):
    """Retrieve relevant text from Pinecone based on query."""
    try:
        query_embedding = embed_model.encode([query]).tolist()[0]
        results = index.query(
            vector=query_embedding,
            top_k=CONFIG["top_k_results"],
            include_metadata=True
        )

        if results['matches']:
            contexts = []
            sources = {}  # Changed to dictionary to track source type and URL/path

            for match in results['matches']:
                # Extract text and source information
                contexts.append(match['metadata']['text'])

                source_id = match['metadata'].get('source', 'unknown')
                source_type = match['metadata'].get('source_type', 'unknown')

                # Track sources with their types and URLs
                if source_id not in sources:
                    sources[source_id] = {
                        'type': source_type,
                        'url': match['metadata'].get('url', '')
                    }

            relevant_text = "\n\n".join(contexts)

            # Format source information based on source type
            source_info = []
            source_types = set()

            for source_id, details in sources.items():
                source_type = details['type']
                source_types.add(source_type)

                if source_type == 'pdf':
                    # For PDFs, create a link to the saved file
                    pdf_path = f"{CONFIG['output_dir']}/{source_id}.pdf"
                    if os.path.exists(pdf_path):
                        source_info.append(f"- PDF: [{source_id}]({pdf_path})")
                    else:
                        source_info.append(f"- PDF: {source_id}")
                elif source_type == 'website' and details['url']:
                    # For websites, link to the original URL
                    source_info.append(f"- Website: [{urlparse(details['url']).netloc}]({details['url']})")
                else:
                    source_info.append(f"- {source_type}: {source_id}")

            formatted_source_info = "\n".join(source_info)
            source_type_info = f"{', '.join(source_types)}"

            return relevant_text, formatted_source_info, source_type_info
        return "No relevant text found.", "", ""
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        logger.error(f"Error querying Pinecone: {e}")
        return f"Error querying Pinecone: {e}", "", ""

def send_to_gemini(query, retrieved_text, source_info):
    """Send query and text to Gemini for response generation."""
    try:
        model = genai.GenerativeModel(CONFIG["model_name"])

        prompt = f"""
        Given the following query and relevant context, generate a comprehensive and accurate response.

        Query: {query}

        Context:
        {retrieved_text}

        Sources:
        {source_info}

        Instructions:
        1. Provide a detailed answer based solely on the information in the context.
        2. If the context doesn't contain relevant information to answer the query,
            state that clearly rather than making up information.
        3. Where appropriate, reference the source of information in your answer.
        4. Format your response in clear, easy-to-read markdown.
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
            return text if text else "", [url]

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
                full_text += "\n\n" + page_text
            time.sleep(CONFIG["sleep_between_requests"])

        return full_text if full_text else "", relevant_links

# Main Application Code (Streamlit UI)

st.title("PDF and Website Content Processing App")

tabs = st.tabs(["Upload PDF", "Enter Website URL"])

with tabs[0]:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        # Reset session state when uploading new file
        reset_session_for_new_source()
        st.session_state.source_type = "pdf"

        # Extract text from PDF
        extracted_text = extract_text_from_pdf(uploaded_file)

        if extracted_text:
            # Generate a unique document ID
            doc_id = f"pdf_{uploaded_file.name.split('.')[0]}_{uuid.uuid4().hex[:8]}"
            st.session_state.extracted_text = extracted_text
            st.session_state.doc_id = doc_id
            st.success(f"Successfully extracted {len(extracted_text)} characters from PDF")

            # Save PDF for reference
            pdf_path = save_pdf_for_reference(uploaded_file)

            # Store embeddings
            if st.button("Process PDF", key="process_pdf"):
                store_embeddings(extracted_text, st.session_state.doc_id, "pdf")

with tabs[1]:
    url = st.text_input("Enter website URL", "https://")

    if url and url.startswith("http"):
        # Reset session state when entering new URL
        reset_session_for_new_source()
        st.session_state.source_type = "website"
        st.session_state.source_url = url

        if st.button("Crawl Website", key="crawl_website"):
            # Crawl the website and extract content
            extracted_text, crawled_urls = crawl_website(url, "company information")

            if extracted_text:
                # Generate a unique document ID
                doc_id = f"web_{urlparse(url).netloc}_{uuid.uuid4().hex[:8]}"
                st.session_state.extracted_text = extracted_text
                st.session_state.doc_id = doc_id
                st.success(f"Successfully extracted content from website")

                # Store embeddings with source URL
                store_embeddings(extracted_text, st.session_state.doc_id, "website", url)

# Querying Section
st.markdown("---")
st.subheader("Ask Questions About the Content")

# Always show the query input box
query = st.text_input("Enter your query:")

# Check if data has been processed or is available in Pinecone
if 'data_processed' in st.session_state and st.session_state.data_processed:
    # Display current source
    if st.session_state.current_source:
        st.info(f"Currently working with: {st.session_state.current_source}")

    if query:
        if st.button("Generate Answer", key="generate"):
            with st.spinner("Retrieving relevant information..."):
                retrieved_text, source_info, source_type_info = retrieve_relevant_text(query)

            if retrieved_text and retrieved_text != "No relevant text found.":
                with st.spinner("Generating AI response..."):
                    response = send_to_gemini(query, retrieved_text, source_info)

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
else:
    # Show a message if no data has been processed yet
    st.info("Please upload a PDF file or crawl a website first to process data for querying.")

    # Add a debug button to check Pinecone index status
    if st.button("Check Database Status"):
        try:
            # Count vectors in index
            stats = index.describe_index_stats()
            vector_count = stats['total_vector_count']

            if vector_count > 0:
                st.success(f"Found {vector_count} vectors in database. Try refreshing the page to enable querying.")
                st.session_state.data_processed = True
            else:
                st.warning("No vectors found in database. Please process a document first.")
        except Exception as e:
            st.error(f"Error checking database: {e}")