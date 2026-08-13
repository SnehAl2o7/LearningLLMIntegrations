import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import API from '../api/axios';

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

  const handleRemove = async (isbn13) => {
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
    <div className="page-enter">
      <h1 className="text-3xl font-bold text-white">My Favorites</h1>
      <p className="text-gray-400 mt-1">Your personally curated book collection</p>
      
      <div className="mt-8">
        {loading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-5">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="rounded-xl overflow-hidden bg-white/[0.03] border border-white/[0.06] animate-pulse">
                <div className="aspect-[2/3] bg-white/5"></div>
                <div className="p-3">
                  <div className="h-4 bg-white/10 rounded w-3/4 mb-2"></div>
                  <div className="h-3 bg-white/5 rounded w-1/2"></div>
                </div>
              </div>
            ))}
          </div>
        ) : favorites.length === 0 ? (
          <div className="glass-card p-12 text-center max-w-md mx-auto mt-12">
            <svg className="w-16 h-16 text-gray-600 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" className="text-red-500/50 fill-red-500/10"></path>
            </svg>
            <h2 className="text-xl font-semibold text-gray-300 mt-4">No favorites yet</h2>
            <p className="text-gray-500 mt-2">Discover books and save your favorites here</p>
            <Link to="/discover" className="btn-primary inline-block px-6 py-2.5 rounded-lg mt-6 font-medium text-sm text-white">
              Start Discovering →
            </Link>
          </div>
        ) : (
          <>
            <p className="text-sm text-gray-500 mb-4">{favorites.length} books saved</p>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-5">
              {favorites.map(fav => (
                <div key={fav.isbn13} className="fav-card group rounded-xl overflow-hidden bg-white/[0.03] border border-white/[0.06] relative">
                  <div className="aspect-[2/3] w-full">
                    {fav.thumbnail ? (
                      <img src={fav.thumbnail} alt={fav.title} className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-gray-800 text-gray-600">No Cover</div>
                    )}
                  </div>
                  <button
                    onClick={() => handleRemove(fav.isbn13)}
                    className="remove-btn absolute top-2 right-2 w-7 h-7 rounded-full bg-red-500/80 hover:bg-red-500 flex items-center justify-center text-white transition-colors"
                  >
                    {removingId === fav.isbn13 ? (
                      <svg className="w-4 h-4 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                      </svg>
                    )}
                  </button>
                  <div className="p-3">
                    <p className="text-sm font-medium text-gray-200 line-clamp-2">{fav.title}</p>
                    <p className="text-xs text-gray-500 mt-1 truncate">{fav.authors || 'Unknown Author'}</p>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
