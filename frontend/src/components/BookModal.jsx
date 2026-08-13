import { useState } from 'react';
import API from '../api/axios';

export default function BookModal({ book, onClose }) {
  const [adding, setAdding] = useState(false);
  const [feedback, setFeedback] = useState(null); // { type: 'success' | 'error' | 'info', msg }
  const [imgError, setImgError] = useState(false);

  const coverUrl =
    !imgError && book.large_thumbnail
      ? book.large_thumbnail
      : !imgError && book.thumbnail
        ? book.thumbnail
        : null;

  const addToFavorites = async () => {
    setAdding(true);
    setFeedback(null);
    try {
      await API.post('/favorites/add/', {
        isbn13: book.isbn13,
        title: book.title || '',
        authors: book.authors || '',
        thumbnail: book.thumbnail || '',
      });
      setFeedback({ type: 'success', msg: 'Added to favorites!' });
    } catch (err) {
      if (err.response?.status === 409) {
        setFeedback({ type: 'info', msg: 'Already in your favorites' });
      } else {
        setFeedback({
          type: 'error',
          msg: err.response?.data?.detail || 'Failed to add',
        });
      }
    } finally {
      setAdding(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 modal-backdrop"
      style={{ background: 'rgba(0, 0, 0, 0.7)' }}
      onClick={onClose}
    >
      <div
        className="modal-content glass-card w-full max-w-2xl max-h-[90vh] overflow-y-auto p-6 sm:p-8"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-gray-400 hover:text-white transition-colors"
          aria-label="Close"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Content */}
        <div className="flex flex-col sm:flex-row gap-6">
          {/* Book Cover */}
          <div className="flex-shrink-0 mx-auto sm:mx-0">
            {coverUrl ? (
              <img
                src={coverUrl}
                alt={book.title}
                className="w-40 h-60 object-cover rounded-xl shadow-2xl shadow-purple-500/10"
                onError={() => setImgError(true)}
              />
            ) : (
              <div className="w-40 h-60 rounded-xl bg-gradient-to-br from-purple-900/30 to-gray-800 flex items-center justify-center">
                <svg className="w-16 h-16 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                </svg>
              </div>
            )}
          </div>

          {/* Details */}
          <div className="flex-1 min-w-0">
            <h2 className="text-xl sm:text-2xl font-bold text-white leading-tight">
              {book.title || 'Untitled'}
            </h2>

            <p className="text-purple-400 mt-2 font-medium">
              {book.authors || 'Unknown Author'}
            </p>

            {/* Meta row */}
            <div className="flex flex-wrap items-center gap-3 mt-3">
              {book.simple_categories && (
                <span className="px-2.5 py-1 rounded-full bg-purple-500/15 text-purple-300 text-xs font-medium border border-purple-500/20">
                  {book.simple_categories}
                </span>
              )}
              {book.average_rating && (
                <span className="flex items-center gap-1 text-sm">
                  <span className="text-amber-400">★</span>
                  <span className="text-gray-300">
                    {Number(book.average_rating).toFixed(1)}
                  </span>
                </span>
              )}
              {book.num_pages && (
                <span className="text-xs text-gray-500">
                  {Math.round(book.num_pages)} pages
                </span>
              )}
              {book.published_year && (
                <span className="text-xs text-gray-500">
                  {Math.round(book.published_year)}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Description */}
        {book.description && (
          <div className="mt-6">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-2">
              Description
            </h3>
            <p className="text-gray-300 text-sm leading-relaxed max-h-48 overflow-y-auto pr-2">
              {book.description}
            </p>
          </div>
        )}

        {/* Feedback message */}
        {feedback && (
          <div
            className={`toast mt-4 px-4 py-2.5 rounded-lg text-sm font-medium ${
              feedback.type === 'success'
                ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20'
                : feedback.type === 'info'
                  ? 'bg-blue-500/15 text-blue-400 border border-blue-500/20'
                  : 'bg-red-500/15 text-red-400 border border-red-500/20'
            }`}
          >
            {feedback.type === 'success' && '✓ '}
            {feedback.type === 'info' && 'ℹ '}
            {feedback.type === 'error' && '✕ '}
            {feedback.msg}
          </div>
        )}

        {/* Add to favorites button */}
        <button
          onClick={addToFavorites}
          disabled={adding || feedback?.type === 'success'}
          className={`mt-5 w-full py-3 rounded-xl font-semibold text-sm transition-all duration-200 flex items-center justify-center gap-2 ${
            feedback?.type === 'success'
              ? 'bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 cursor-default'
              : 'btn-primary text-white disabled:opacity-50'
          }`}
        >
          {adding ? (
            <>
              <div className="flex gap-1">
                <div className="w-1.5 h-1.5 rounded-full bg-white loading-dot" />
                <div className="w-1.5 h-1.5 rounded-full bg-white loading-dot" />
                <div className="w-1.5 h-1.5 rounded-full bg-white loading-dot" />
              </div>
              Adding...
            </>
          ) : feedback?.type === 'success' ? (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
              </svg>
              Added to Favorites
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z" />
              </svg>
              Add to Favorites
            </>
          )}
        </button>
      </div>
    </div>
  );
}
