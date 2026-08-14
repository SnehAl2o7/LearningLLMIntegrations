import React, { useState, useRef, useEffect } from 'react';
import API from '../api/axios';
import BookCard from '../components/BookCard';
import BookModal from '../components/BookModal';
import LoadingSkeleton from '../components/LoadingSkeleton';

const CATEGORIES = ['All', 'Fiction', 'Nonfiction', 'Science', 'History', 'Biography', 'Fantasy', 'Romance', 'Mystery', 'Self-Help'];
const TONES = ['All', 'Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad'];

const SUGGESTIONS = [
  'A thrilling mystery set in Victorian London',
  'A heartwarming romance in a small coastal town',
  'An epic fantasy with dragons and magic',
  'A mind-bending sci-fi about time travel',
  'A gripping psychological thriller',
  'A beautiful literary fiction about family secrets',
  'An inspiring biography of a visionary leader',
  'A cozy mystery with an amateur detective',
];

export default function DiscoverPage() {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('All');
  const [tone, setTone] = useState('All');
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedBook, setSelectedBook] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [focused, setFocused] = useState(false);
  const inputRef = useRef(null);
  const suggestionsRef = useRef(null);

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (inputRef.current && !inputRef.current.contains(e.target) &&
          suggestionsRef.current && !suggestionsRef.current.contains(e.target)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setShowSuggestions(false);
    
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

  const handleSuggestionClick = (suggestion) => {
    setQuery(suggestion);
    setShowSuggestions(false);
    handleSearch();
  };

  const filteredSuggestions = SUGGESTIONS.filter(s => 
    s.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="page-enter min-h-screen">
      {/* Hero Section */}
      <section className="relative py-16 sm:py-24 px-4 sm:px-6 lg:px-8 hero-pattern">
        <div className="max-w-7xl mx-auto">
          <div className="text-center max-w-3xl mx-auto">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-300 text-sm font-medium mb-6 scale-in">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
              </span>
              Semantic Search Powered by AI
            </div>
            
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white leading-tight mb-4 gradient-text">
              Discover Your Next{' '}
              <span className="gradient-text-cyan">Favorite Book</span>
            </h1>
            
            <p className="text-lg sm:text-xl text-gray-300 mb-8 max-w-2xl mx-auto leading-relaxed">
              Describe what you're in the mood for — a mood, a setting, a character — and let our AI find the perfect match from thousands of titles.
            </p>

            {/* Search Card */}
            <div className="glass-card-elevated p-6 sm:p-8 max-w-3xl mx-auto glow-pulse">
              <div className="relative">
                <div className="relative">
                  <label htmlFor="discover-search-input" className="sr-only">Search for books</label>
                  <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-transparent to-pink-500/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    <input
                      ref={inputRef}
                      type="text"
                      id="discover-search-input"
                      value={query}
                      onChange={(e) => {
                        setQuery(e.target.value);
                        setShowSuggestions(true);
                      }}
                      onFocus={() => {
                        setFocused(true);
                        if (query.trim()) setShowSuggestions(true);
                      }}
                      onBlur={() => {
                        setFocused(false);
                        // Delay to allow clicking suggestions
                        setTimeout(() => setShowSuggestions(false), 200);
                      }}
                      onKeyDown={handleKeyDown}
                      placeholder="A thrilling mystery set in Victorian London..."
                      className="w-full bg-white/3 border border-white/10 rounded-2xl px-6 py-5 text-white text-lg placeholder-gray-500 input-glow focus:border-purple-500 focus:outline-none transition-all duration-300 pr-16"
                      autoComplete="off"
                    />
                    {query && (
                      <button
                        onClick={() => setQuery('')}
                        className="absolute right-14 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors p-1"
                        aria-label="Clear search"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>

                {/* Suggestions Dropdown */}
                {showSuggestions && (focused || query.trim()) && filteredSuggestions.length > 0 && (
                  <div
                    ref={suggestionsRef}
                    className="absolute top-full left-0 right-0 mt-2 glass-card-elevated rounded-xl overflow-hidden shadow-2xl z-20 animate-slideUp"
                  >
                    <div className="px-4 py-2 border-b border-white/10">
                      <p className="text-xs text-gray-500 font-medium uppercase tracking-wider">Suggestions</p>
                    </div>
                    <div className="py-2 max-h-60 overflow-y-auto">
                      {filteredSuggestions.map((suggestion, index) => (
                        <button
                          key={suggestion}
                          onClick={() => handleSuggestionClick(suggestion)}
                          className="w-full text-left px-4 py-3 rounded-lg hover:bg-white/5 transition-colors flex items-center gap-3 group"
                        >
                          <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                            <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                          </div>
                          <span className="text-gray-200 text-sm">{suggestion}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Filters */}
              <div className="mt-6 flex flex-col sm:flex-row gap-3 items-stretch sm:items-center">
                <div className="flex-1 sm:w-48">
                  <label htmlFor="discover-category-select" className="sr-only">Category</label>
                  <select
                    id="discover-category-select"
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    className="select-styled w-full bg-white/3 border border-white/10 rounded-xl px-4 py-3 text-sm text-gray-300 focus:outline-none focus:border-purple-500 transition-colors appearance-none cursor-pointer"
                  >
                    {CATEGORIES.map(cat => (
                      <option key={cat} value={cat}>{cat}</option>
                    ))}
                  </select>
                </div>
                
                <div className="flex-1 sm:w-48">
                  <label htmlFor="discover-tone-select" className="sr-only">Tone</label>
                  <select
                    id="discover-tone-select"
                    value={tone}
                    onChange={(e) => setTone(e.target.value)}
                    className="select-styled w-full bg-white/3 border border-white/10 rounded-xl px-4 py-3 text-sm text-gray-300 focus:outline-none focus:border-purple-500 transition-colors appearance-none cursor-pointer"
                  >
                    {TONES.map(t => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>
                
                <button
                  id="discover-search-btn"
                  onClick={handleSearch}
                  disabled={loading || !query.trim()}
                  className="btn-primary px-8 py-3.5 rounded-xl font-semibold text-sm text-white disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 w-full sm:w-auto"
                  aria-busy={loading}
                >
                  {loading ? (
                    <>
                      <div className="flex gap-1 items-center">
                        <div className="w-1.5 h-1.5 bg-white rounded-full loading-dot"></div>
                        <div className="w-1.5 h-1.5 bg-white rounded-full loading-dot"></div>
                        <div className="w-1.5 h-1.5 bg-white rounded-full loading-dot"></div>
                      </div>
                      <span>Searching...</span>
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                      <span>Discover Books</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Quick suggestions when empty */}
            {!hasSearched && !loading && books.length === 0 && (
              <div className="mt-8">
                <p className="text-sm text-gray-500 mb-4 text-center">Or try one of these:</p>
                <div className="flex flex-wrap justify-center gap-2 max-w-3xl mx-auto">
                  {SUGGESTIONS.slice(0, 6).map((suggestion, index) => (
                    <button
                      key={suggestion}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className="btn-secondary px-4 py-2 rounded-full text-sm text-gray-300 hover:text-white whitespace-nowrap stagger-1"
                      style={{ animationDelay: `${index * 50}ms` }}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Results Section */}
      <section className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          {loading ? (
            <LoadingSkeleton count={16} />
          ) : hasSearched && books.length === 0 ? (
            <div className="empty-state glass-card max-w-md mx-auto mt-12">
              <div className="empty-state-icon">
                <svg className="w-16 h-16 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-gray-200">No books found</h2>
              <p className="text-gray-500 mt-2">Try a different description or adjust your filters</p>
              <button
                onClick={() => { setQuery(''); setHasSearched(false); }}
                className="btn-primary inline-block px-6 py-2.5 rounded-lg mt-6 font-medium text-sm text-white"
              >
                Clear & Try Again
              </button>
            </div>
          ) : books.length > 0 ? (
            <>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <p className="text-sm text-gray-400">
                    <span className="text-white font-semibold">{books.length}</span> recommendations found
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    For: <span className="text-purple-300">"{query}"</span>
                    {category !== 'All' && <span className="mx-1">•</span>}
                    {category !== 'All' && <span className="text-cyan-300">{category}</span>}
                    {tone !== 'All' && <span className="mx-1">•</span>}
                    {tone !== 'All' && <span className="text-pink-300">{tone}</span>}
                  </p>
                </div>
                <button
                  onClick={() => { setQuery(''); setHasSearched(false); setBooks([]); }}
                  className="btn-secondary px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white"
                >
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  New Search
                </button>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-5">
                {books.map((book, index) => (
                  <BookCard
                    key={book.isbn13}
                    book={book}
                    onClick={setSelectedBook}
                    className={`stagger-${Math.min(index + 1, 16)}`}
                  />
                ))}
              </div>
            </>
          ) : null}
        </div>
      </section>

      {/* Modal */}
      {selectedBook && (
        <BookModal book={selectedBook} onClose={() => setSelectedBook(null)} />
      )}
    </div>
  );
}
