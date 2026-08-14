import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import API from '../api/axios';
import BookCard from '../components/BookCard';
import LoadingSkeleton from '../components/LoadingSkeleton';

export default function FavoritesPage() {
  const [favorites, setFavorites] = useState([]);
  const [loading, setLoading] = useState(true);
  const [removingId, setRemovingId] = useState(null);

  useEffect(() => {
    const fetchFavorites = async () => {
      try {
        const response = await API.get('/favorites/');
        setFavorites(response.data.favorites);
      } catch (error) {
        console.error('Failed to fetch favorites:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchFavorites();
  }, []);

  const handleRemove = async (isbn13, e) => {
    e.stopPropagation();
    setRemovingId(isbn13);
    try {
      await API.delete(`/favorites/remove/${isbn13}/`);
      setFavorites(favorites.filter(fav => fav.isbn13 !== isbn13));
    } catch (error) {
      console.error('Failed to remove favorite:', error);
    } finally {
      setRemovingId(null);
    }
  };

  return (
    <div className="page-enter min-h-screen">
      {/* Hero Section */}
      <section className="relative py-12 sm:py-16 px-4 sm:px-6 lg:px-8 hero-pattern">
        <div className="max-w-7xl mx-auto">
          <div className="text-center max-w-3xl mx-auto">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-pink-500/10 border border-pink-500/20 text-pink-300 text-sm font-medium mb-6 scale-in">
              <svg className="w-4 h-4 animate-pulse" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
              </svg>
              Your Personal Library
            </div>
            
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white leading-tight mb-4 gradient-text">
              My <span className="gradient-text-pink">Favorites</span>
            </h1>
            
            <p className="text-lg sm:text-xl text-gray-300 mb-8 max-w-2xl mx-auto leading-relaxed">
              Your curated collection of books you love. Revisit old favorites or discover them again.
            </p>
          </div>
        </div>
      </section>

      {/* Results Section */}
      <section className="py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          {loading ? (
            <LoadingSkeleton count={16} />
          ) : favorites.length === 0 ? (
            <div className="empty-state glass-card max-w-md mx-auto mt-12">
              <div className="empty-state-icon">
                <svg className="w-16 h-16 text-gray-500" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-gray-200">No favorites yet</h2>
              <p className="text-gray-500 mt-2">Discover books and save your favorites here</p>
              <Link to="/discover" className="btn-primary inline-block px-6 py-2.5 rounded-lg mt-6 font-medium text-sm text-white">
                Start Discovering →
              </Link>
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-6">
                <div>
                  <p className="text-sm text-gray-400">
                    <span className="text-white font-semibold">{favorites.length}</span> {favorites.length === 1 ? 'book saved' : 'books saved'}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-5">
                {favorites.map((fav, index) => (
                  <div key={fav.isbn13} className={`stagger-${Math.min(index + 1, 16)}`}>
                    <BookCard
                      book={fav}
                      onClick={() => {}}
                      className="relative"
                    >
                      <button
                        onClick={(e) => handleRemove(fav.isbn13, e)}
                        disabled={removingId === fav.isbn13}
                        className="absolute top-2 right-2 z-10 w-8 h-8 rounded-full bg-red-500/80 hover:bg-red-500 flex items-center justify-center text-white transition-all duration-200 hover:scale-110 disabled:opacity-50 disabled:cursor-wait"
                        aria-label="Remove from favorites"
                      >
                        {removingId === fav.isbn13 ? (
                          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                        ) : (
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        )}
                      </button>
                    </BookCard>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
