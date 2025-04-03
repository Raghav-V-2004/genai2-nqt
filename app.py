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
import uuid
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
 
# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("GEMINI_API_KEY not found in .env file!")
   
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
st.set_page_config(
    page_title="Content Assistant",
    page_icon="📄",
    layout="wide"
)
# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'doc_id' not in st.session_state:
    st.session_state.doc_id = ""
if 'source_type' not in st.session_state:
    st.session_state.source_type = ""
if 'source_url' not in st.session_state:
    st.session_state.source_url = ""
if 'source_name' not in st.session_state:
    st.session_state.source_name = ""
if 'source_links' not in st.session_state:
    st.session_state.source_links = {}  # Dictionary to store source links
if 'namespace' not in st.session_state:
    st.session_state.namespace = ""  # For context isolation in Pinecone
 
# Load embedding model
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
 
embed_model = get_embedding_model()
 
# Initialize Pinecone client
@st.cache_resource
def init_pinecone():
    if not PINECONE_API_KEY:
        st.error("PINECONE_API_KEY not found in .env file!")
        return None
        
    try:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        index_name = "newindex"
        
        # Check if index exists, create if it doesn't
        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name not in existing_indexes:
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
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        logger.error(f"Error initializing Pinecone: {e}")
        return None
 
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

def create_new_session():
    """Create a new session and reset state variables."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.data_processed = False
    st.session_state.extracted_text = ""
    st.session_state.doc_id = ""
    st.session_state.source_type = ""
    st.session_state.source_url = ""
    st.session_state.source_name = ""
    st.session_state.source_links = {}
    st.session_state.namespace = f"ns_{st.session_state.session_id[:8]}"
    
    # Clear previous embeddings for this session if they exist
    if index:
        try:
            index.delete(delete_all=True, namespace=st.session_state.namespace)
            logger.info(f"Cleared previous embeddings for namespace {st.session_state.namespace}")
        except Exception as e:
            logger.warning(f"No previous embeddings to clear or error: {e}")

def generate_chunk_id(doc_id, chunk_index):
    """Generate a unique ID for each text chunk."""
    return f"{doc_id}_{chunk_index}"

def chunk_text(text, max_chunk_size=1000, overlap=100):
    """Split text into manageable chunks with overlap."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + max_chunk_size, text_length)
        
        # If not at the end of text, try to find a sentence boundary
        if end < text_length:
            # Find the last period, question mark, or exclamation point
            last_period = max(
                text.rfind('. ', start, end),
                text.rfind('? ', start, end),
                text.rfind('! ', start, end)
            )
            
            if last_period != -1:
                end = last_period + 1  # Include the period
        
        chunks.append(text[start:end].strip())
        start = end - overlap if end < text_length else end
        
    return chunks
 
def store_embeddings(text, doc_id, source_type, source_url="", source_name=""):
    """Store text embeddings in Pinecone with metadata using current namespace."""
    if not index:
        st.error("Pinecone index is not initialized.")
        return False
        
    try:
        # Update source tracking
        if source_url and source_url not in st.session_state.source_links:
            st.session_state.source_links[source_url] = source_name or source_url
            
        # Split text into manageable chunks
        chunks = chunk_text(text)
 
        if not chunks:
            st.warning("No text to embed.")
            return False
 
        # Create embeddings
        with st.spinner("Creating embeddings..."):
            all_vectors = []
            
            # Process chunks in batches to avoid memory issues
            for i, chunk in enumerate(chunks):
                chunk_id = generate_chunk_id(doc_id, i)
                embedding = embed_model.encode([chunk]).tolist()[0]
                
                # Create vector with metadata
                vector = (chunk_id, embedding, {
                    "text": chunk,
                    "source": doc_id,
                    "source_type": source_type,
                    "source_url": source_url,
                    "source_name": source_name or doc_id
                })
                
                all_vectors.append(vector)
 
        # Upload to Pinecone in batches
        batch_size = 100
        total_vectors = len(all_vectors)
 
        # Initialize the progress bar
        progress_bar = st.progress(0)
 
        for i in range(0, total_vectors, batch_size):
            end_idx = min(i + batch_size, total_vectors)
            batch = all_vectors[i:end_idx]
            
            # Use the session-specific namespace
            index.upsert(vectors=batch, namespace=st.session_state.namespace)
           
            # Update progress
            progress_bar.progress((end_idx) / total_vectors)
 
        st.success(f"Successfully stored {len(all_vectors)} text chunks in the database!")
        st.session_state.data_processed = True
        return True
        
    except Exception as e:
        st.error(f"Error storing embeddings: {e}")
        logger.error(f"Error storing embeddings: {e}")
        return False
 
def retrieve_relevant_text(query):
    """Retrieve relevant text from Pinecone based on query within current namespace."""
    if not index:
        st.error("Pinecone index is not initialized.")
        return "Database connection error.", "", ""
        
    try:
        query_embedding = embed_model.encode([query]).tolist()[0]
        results = index.query(
            vector=query_embedding, 
            top_k=5, 
            include_metadata=True,
            namespace=st.session_state.namespace
        )
       
        if results['matches']:
            contexts = []
            sources = {}  # Dictionary to track sources with their URLs
           
            for match in results['matches']:
                if match['score'] < 0.6:  # Relevance threshold
                    continue
                    
                # Get text and source information
                if 'metadata' in match and 'text' in match['metadata']:
                    contexts.append(match['metadata']['text'])
                    
                    # Track source with URL if available
                    if 'source_url' in match['metadata'] and match['metadata']['source_url']:
                        source_key = match['metadata']['source']
                        sources[source_key] = {
                            'url': match['metadata']['source_url'],
                            'name': match['metadata']['source_name'],
                            'type': match['metadata']['source_type']
                        }
                    elif 'source' in match['metadata']:
                        source_key = match['metadata']['source']
                        sources[source_key] = {
                            'name': match['metadata']['source_name'],
                            'type': match['metadata']['source_type']
                        }
           
            if not contexts:
                return "No sufficiently relevant text found.", "", ""
                
            relevant_text = "\n\n".join(contexts)
           
            # Format source info for display
            source_links = []
            source_types = set()
            
            for source_key, source_data in sources.items():
                source_types.add(source_data.get('type', ''))
                
                if source_data.get('type') == 'pdf':
                    source_name = source_data.get('name', source_key)
                    if source_data.get('url'):
                        source_links.append(f"[PDF: {source_name}]({source_data['url']})")
                    else:
                        source_links.append(f"PDF: {source_name}")
                elif source_data.get('url'):
                    source_name = source_data.get('name', source_data['url'])
                    source_links.append(f"[Web: {source_name}]({source_data['url']})")
                else:
                    source_links.append(f"Source: {source_data.get('name', source_key)}")
           
            source_info = "\n".join(source_links)
            source_type_info = ", ".join(source_types)
           
            return relevant_text, source_info, source_type_info
        
        return "No relevant text found.", "", ""
        
    except Exception as e:
        st.error(f"Error querying database: {e}")
        logger.error(f"Error querying database: {e}")
        return f"Error querying database: {e}", "", ""
 
def send_to_gemini(query, retrieved_text):
    """Send query and text to Gemini for response generation."""
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."
        
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
            clean_content = clean_text(text)
            
            # Save PDF file for future reference
            pdf_path = os.path.join(CONFIG["output_dir"], f"{pdf_file.name}")
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())
                
            return clean_content, pdf_path
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        logger.error(f"Error extracting text from PDF: {e}")
        return "", ""
 
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
        
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not configured, returning first 5 links")
        return list(links)[:5]
       
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
            return list(links)[:5]  # Just return the first 5 links
    except Exception as e:
        logger.error(f"Error in filtering links: {e}")
        return list(links)[:5]  # Just return the first 5 links
 
 
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
            return text if text else "", {url: "Main Page"}
       
        # Filter links to find the most relevant ones based on query
        relevant_links = filter_relevant_links(list(all_links), query)
       
        if not relevant_links:
            st.warning("No relevant pages identified, using main page")
            relevant_links = [url]
        else:
            st.success(f"Found {len(relevant_links)} relevant pages")
   
    # Extract text from relevant pages
    full_text = ""
    page_links = {}
    
    for link in relevant_links:
        page_text = extract_text_from_url(link)
        if page_text:
            full_text += page_text
            # Store the link with a page title (using URL as fallback)
            parsed_url = urlparse(link)
            page_name = parsed_url.path.split('/')[-1]
            if not page_name:
                page_name = parsed_url.netloc
            page_links[link] = page_name.replace('-', ' ').replace('_', ' ').capitalize()
            
        time.sleep(CONFIG["sleep_between_requests"])
   
    return full_text if full_text else "", page_links
 
# Main Application Code (Streamlit UI)
 


# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4682b4;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        padding: 20px;
        margin-bottom: 20px;
    }
    .response-area {
        background-color: #f0f8ff;
        border-left: 5px solid #4682b4;
        padding: 15px;
        margin-top: 10px;
    }
    .source-link {
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Content Assistant</h1>", unsafe_allow_html=True)

# Create a sidebar for session management
with st.sidebar:
    st.header("Session Controls")
    if st.button("Start New Session"):
        create_new_session()
        st.success("Created new session!")
    
    st.divider()
    st.write(f"Current Session ID: {st.session_state.session_id[:8]}...")
    if st.session_state.source_name:
        st.write(f"Current Source: {st.session_state.source_name}")
    elif st.session_state.doc_id:
        st.write(f"Current Source ID: {st.session_state.doc_id}")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h2 class='sub-header'>Upload Content</h2>", unsafe_allow_html=True)
    
    tabs = st.tabs(["PDF Document", "Website URL"])
    
    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        
        if uploaded_file:
            # Reset previous data when uploading new file
            if st.session_state.doc_id and not st.session_state.doc_id.startswith(f"pdf_{uploaded_file.name}"):
                if st.button("Process New PDF", help="This will replace your current session data"):
                    create_new_session()
                    
            # Only show process button when we have a new file
            if not st.session_state.data_processed or st.session_state.doc_id != f"pdf_{uploaded_file.name}":
                if st.button("Process PDF", key="process_pdf"):
                    # Extract text from PDF
                    extracted_text, pdf_path = extract_text_from_pdf(uploaded_file)
            
                    if extracted_text:
                        st.session_state.extracted_text = extracted_text
                        st.session_state.doc_id = f"pdf_{uploaded_file.name}"
                        st.session_state.source_type = "pdf"
                        st.session_state.source_url = pdf_path
                        st.session_state.source_name = uploaded_file.name
                        
                        # Store embeddings
                        success = store_embeddings(
                            extracted_text, 
                            st.session_state.doc_id, 
                            "pdf",
                            pdf_path,  # URL is the local path
                            uploaded_file.name
                        )
                        
                        if success:
                            st.success(f"Successfully processed PDF with {len(extracted_text)} characters")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        url = st.text_input("Enter website URL", "https://")
        
        if url and url != "https://":
            # Reset previous data when entering new URL
            if st.session_state.doc_id and not st.session_state.doc_id.startswith(f"web_{urlparse(url).netloc}"):
                if st.button("Crawl New Website", help="This will replace your current session data"):
                    create_new_session()
            
            # Only show crawl button when we have a new URL
            if not st.session_state.data_processed or st.session_state.doc_id != f"web_{urlparse(url).netloc}":
                query = st.text_input("Enter a topic to focus crawling on (e.g., 'product information')", "company information")
                
                if st.button("Crawl Website", key="crawl_website"):
                    # Crawl the website and extract content
                    extracted_text, page_links = crawl_website(url, query)
        
                    if extracted_text:
                        st.session_state.extracted_text = extracted_text
                        st.session_state.doc_id = f"web_{urlparse(url).netloc}"
                        st.session_state.source_type = "website"
                        st.session_state.source_url = url
                        st.session_state.source_name = urlparse(url).netloc
                        st.session_state.source_links.update(page_links)
                        
                        # Store embeddings
                        success = store_embeddings(
                            extracted_text, 
                            st.session_state.doc_id, 
                            "website",
                            url,
                            urlparse(url).netloc
                        )
                        
                        if success:
                            st.success(f"Successfully processed website content")
                            for link, title in page_links.items():
                                st.markdown(f"- [{title}]({link})")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Only show querying section after data is processed
    if st.session_state.data_processed:
        st.markdown("<h2 class='sub-header'>Ask Questions</h2>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        query = st.text_area("Enter your question about the content:", height=100)
        
        if query:
            if st.button("Generate Answer", key="generate"):
                with st.spinner("Retrieving relevant information..."):
                    retrieved_text, source_info, source_type_info = retrieve_relevant_text(query)
                
                if retrieved_text and retrieved_text != "No relevant text found.":
                    with st.spinner("Generating AI response..."):
                        response = send_to_gemini(query, retrieved_text)
                
                    # Display results
                    st.markdown("<div class='response-area'>", unsafe_allow_html=True)
                    st.markdown("### Answer:")
                    st.markdown(response)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                    # Show sources
                    if source_info:
                        st.markdown("<div class='source-link'>", unsafe_allow_html=True)
                        st.markdown("#### Sources:")
                        st.markdown(source_info)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                    # Optionally show context
                    with st.expander("Show retrieved context"):
                        st.markdown(retrieved_text)
                else:
                    st.warning("No relevant information found for your query. Try a different question.")
                    
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 class='sub-header'>Get Started</h2>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
        To start using the content assistant:
        
        1. First, upload a PDF document or enter a website URL
        2. Process the content to extract information
        3. Once processing is complete, you can ask questions about the content""")
