import { useState } from 'react';

export default function BookCard({ book, onClick, className = '' }) {
  const [imgError, setImgError] = useState(false);

  const coverUrl =
    !imgError && book.large_thumbnail
      ? book.large_thumbnail
      : !imgError && book.thumbnail
        ? book.thumbnail
        : null;

  return (
    <div
      onClick={() => onClick(book)}
      className={`book-card cursor-pointer group rounded-xl overflow-hidden bg-white/[0.03] border border-white/[0.06] hover:border-purple-500/30 ${className}`}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onClick(book)}
      id={`book-${book.isbn13}`}
    >
      {/* Cover Image */}
      <div className="relative aspect-[2/3] overflow-hidden bg-gradient-to-br from-purple-900/20 to-gray-900/40">
        {coverUrl ? (
          <img
            src={coverUrl}
            alt={book.title || 'Book cover'}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
            onError={() => setImgError(true)}
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center text-gray-500 p-4">
            <svg
              className="w-12 h-12 mb-2 opacity-40"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"
              />
            </svg>
            <span className="text-xs text-center opacity-60">No Cover</span>
          </div>
        )}

        {/* Hover overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end p-3">
          <span className="text-xs text-white/80 font-medium">View Details</span>
        </div>

        {/* Rating badge */}
        {book.average_rating && (
          <div className="absolute top-2 right-2 flex items-center gap-1 px-2.5 py-1 rounded-full bg-black/70 backdrop-blur-sm text-xs border border-white/10">
            <span className="text-amber-400 star-filled">★</span>
            <span className="text-white/90 font-medium">
              {Number(book.average_rating).toFixed(1)}
            </span>
          </div>
        )}

        {/* Category badge */}
        {book.simple_categories && (
          <div className="absolute top-2 left-2 badge badge-purple">
            {book.simple_categories}
          </div>
        )}
      </div>

      {/* Title */}
      <div className="p-4">
        <h3 className="text-sm font-medium text-gray-100 line-clamp-2 leading-snug group-hover:text-purple-300 transition-colors duration-200">
          {book.title || 'Untitled'}
        </h3>
        <p className="text-xs text-gray-500 mt-1.5 truncate group-hover:text-gray-400 transition-colors duration-200">
          {book.authors || 'Unknown Author'}
        </p>
        
        {/* Meta info */}
        <div className="flex items-center gap-2 mt-2.5 text-xs text-gray-500">
          {book.published_year && (
            <span className="flex items-center gap-1">
              <svg className="w-3 h-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              {Math.round(book.published_year)}
            </span>
          )}
          {book.num_pages && (
            <span className="flex items-center gap-1">
              <svg className="w-3 h-3 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              {Math.round(book.num_pages)}p
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
