import React, { useState } from 'react';
import API from '../api/axios';
import BookCard from '../components/BookCard';
import BookModal from '../components/BookModal';
import LoadingSkeleton from '../components/LoadingSkeleton';

const CATEGORIES = ['All', 'Fiction', 'Nonfiction', 'Science', 'History', 'Biography', 'Fantasy', 'Romance', 'Mystery', 'Self-Help'];
const TONES = ['All', 'Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad'];

export default function DiscoverPage() {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('All');
  const [tone, setTone] = useState('All');
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedBook, setSelectedBook] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    try {
      const response = await API.post('/recommendations/', { query, category, tone });
      setBooks(response.data.recommendations);
      setHasSearched(true);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="page-enter">
      <h1 className="text-3xl font-bold text-white">Discover Books</h1>
      <p className="text-gray-400 mt-1">Find your next favorite read using semantic search</p>
      
      <div className="glass-card p-5 mt-6">
        <input
          type="text"
          id="discover-search-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="A thrilling mystery set in Victorian London..."
          className="w-full bg-white/5 border border-white/10 rounded-xl px-5 py-4 text-white input-glow focus:border-purple-500 focus:outline-none transition-colors"
        />
        
        <div className="flex flex-wrap gap-3 mt-4 items-center">
          <select
            id="discover-category-select"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-sm text-gray-300 focus:outline-none focus:border-purple-500 transition-colors"
          >
            {CATEGORIES.map(cat => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
          
          <select
            id="discover-tone-select"
            value={tone}
            onChange={(e) => setTone(e.target.value)}
            className="bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-sm text-gray-300 focus:outline-none focus:border-purple-500 transition-colors"
          >
            {TONES.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          
          <button
            id="discover-search-btn"
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="btn-primary px-8 py-2.5 rounded-lg font-semibold text-sm text-white disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <div className="flex gap-1 items-center">
                  <div className="w-1.5 h-1.5 bg-white rounded-full animate-bounce"></div>
                  <div className="w-1.5 h-1.5 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-1.5 h-1.5 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span>Searching...</span>
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                <span>Search</span>
              </>
            )}
          </button>
        </div>
      </div>
      
      <div className="mt-8">
        {loading ? (
          <LoadingSkeleton />
        ) : hasSearched && books.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-lg text-gray-400">No books found</p>
            <p className="text-gray-500 mt-2">Try different description</p>
          </div>
        ) : books.length > 0 ? (
          <>
            <p className="text-sm text-gray-500 mb-4">{books.length} recommendations</p>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-5">
              {books.map(book => (
                <BookCard key={book.isbn13} book={book} onClick={setSelectedBook} />
              ))}
            </div>
          </>
        ) : null}
      </div>
      
      {selectedBook && (
        <BookModal book={selectedBook} onClose={() => setSelectedBook(null)} />
      )}
    </div>
  );
}
