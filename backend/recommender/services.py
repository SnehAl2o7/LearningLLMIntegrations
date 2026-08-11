"""
Standalone service module for the semantic book recommender.

Adapts the Chroma/LangChain recommendation logic from the LLM-handling
notebooks/dashboard into a reusable service for the Django backend.
"""
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma

# Paths to the LLM-handling data files (relative to the backend directory)
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
PROJECT_ROOT = BASE_DIR.parent  # project root
LLM_HANDLING_DIR = PROJECT_ROOT / 'LLM-handling'

# Load environment variables (NVIDIA_API_KEY, etc.) from the project root .env
load_dotenv(PROJECT_ROOT / '.env')
load_dotenv()

BOOKS_CSV_PATH = LLM_HANDLING_DIR / 'books_with_emotions.csv'
TAGGED_DESCRIPTION_PATH = LLM_HANDLING_DIR / 'tagged_description.txt'

# Global singleton references (lazy-initialized)
_books_df = None
_db_books = None


def _load_books_dataframe() -> pd.DataFrame:
    """Load the books DataFrame once and cache it."""
    global _books_df
    if _books_df is None:
        _books_df = pd.read_csv(BOOKS_CSV_PATH)
        # Add a large thumbnail column for richer responses
        _books_df['large_thumbnail'] = _books_df['thumbnail'] + '&fife=w800'
        _books_df['large_thumbnail'] = _books_df['large_thumbnail'].fillna('cover-not-found.jpg')
    return _books_df


def _get_chroma_db() -> Chroma:
    """
    Build (or reuse) the in-memory Chroma vector store.

    Uses NVIDIAEmbeddings with the nv-embedqa-e5-v5 model and the
    tagged_description.txt file as the document source.
    """
    global _db_books
    if _db_books is None:
        nvidia_api_key = os.getenv('NVIDIA_API_KEY')
        if not nvidia_api_key:
            raise RuntimeError(
                'NVIDIA_API_KEY environment variable is not set. '
                'Please set it in your .env file or environment.'
            )

        # Load and split the tagged descriptions
        raw_documents = TextLoader(str(TAGGED_DESCRIPTION_PATH)).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        documents = text_splitter.split_documents(raw_documents)

        # Create the embedder
        embedder = NVIDIAEmbeddings(
            model='nvidia/nv-embedqa-e5-v5',
            nvidia_api_key=nvidia_api_key,
        )

        # Build the in-memory Chroma vector store
        _db_books = Chroma.from_documents(
            documents,
            embedding=embedder,
        )

    return _db_books


def get_recommendations(
    query: str,
    category: str = 'All',
    tone: str = 'All',
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> list[dict]:
    """
    Retrieve semantic book recommendations from the Chroma vector store.

    Args:
        query: The user's natural-language book description.
        category: Optional category filter (e.g. 'Fiction', 'Nonfiction').
        tone: Optional emotional tone filter ('Happy', 'Surprising',
              'Angry', 'Suspenseful', 'Sad').
        initial_top_k: Number of candidates to pull from Chroma.
        final_top_k: Number of recommendations to return after filtering.

    Returns:
        A list of book dictionaries with full metadata.
    """
    books = _load_books_dataframe()
    db_books = _get_chroma_db()

    # 1. Semantic search over the Chroma vector store
    recs = db_books.similarity_search(query, k=initial_top_k)

    # 2. Extract 13-digit ISBNs from the matched chunks
    books_list = []
    for doc in recs:
        match = re.search(r'\b\d{13}\b', doc.page_content)
        if match:
            books_list.append(int(match.group()))

    # 3. Filter the main DataFrame by the matched ISBNs
    book_recs = books[books['isbn13'].isin(books_list)].head(initial_top_k)

    # 4. Apply category filter
    if category and category != 'All':
        book_recs = book_recs[book_recs['simple_categories'] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # 5. Apply tone-based sorting
    if tone == 'Happy':
        book_recs = book_recs.sort_values(by='joy', ascending=False)
    elif tone == 'Surprising':
        book_recs = book_recs.sort_values(by='surprise', ascending=False)
    elif tone == 'Angry':
        book_recs = book_recs.sort_values(by='anger', ascending=False)
    elif tone == 'Suspenseful':
        book_recs = book_recs.sort_values(by='fear', ascending=False)
    elif tone == 'Sad':
        book_recs = book_recs.sort_values(by='sadness', ascending=False)

    # 6. Convert to a list of dicts for JSON serialization
    recommendations = []
    for _, row in book_recs.iterrows():
        recommendations.append({
            'isbn13': str(row['isbn13']),
            'isbn10': str(row.get('isbn10', '')),
            'title': row.get('title', ''),
            'authors': row.get('authors', ''),
            'categories': row.get('categories', ''),
            'simple_categories': row.get('simple_categories', ''),
            'thumbnail': row.get('thumbnail', ''),
            'large_thumbnail': row.get('large_thumbnail', ''),
            'description': row.get('description', ''),
            'published_year': row.get('published_year'),
            'average_rating': row.get('average_rating'),
            'num_pages': row.get('num_pages'),
            'ratings_count': row.get('ratings_count'),
            'title_and_subtitle': row.get('title_and_subtitle', ''),
        })

    return recommendations


def get_available_categories() -> list[str]:
    """Return the sorted list of available book categories."""
    books = _load_books_dataframe()
    return ['All'] + sorted(books['simple_categories'].dropna().unique().tolist())


def get_available_tones() -> list[str]:
    """Return the list of available emotional tones."""
    return ['All', 'Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad']