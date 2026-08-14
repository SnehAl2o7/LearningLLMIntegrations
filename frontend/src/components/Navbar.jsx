import { NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Navbar() {
  const { user, logout } = useAuth();

  return (
    <nav className="navbar-glass sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <NavLink to="/discover" className="flex items-center gap-3 group">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-amber-400 rounded-full animate-ping opacity-75" />
            </div>
            <h1 className="text-xl font-bold gradient-text tracking-tight group-hover:scale-105 transition-transform duration-300">
              BookLens
            </h1>
          </NavLink>

          {/* Navigation Links */}
          <div className="flex items-center gap-1">
            <NavLink
              to="/discover"
              className={({ isActive }) =>
                `relative px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
                  isActive
                    ? 'bg-gradient-to-r from-purple-600/90 to-pink-600/90 text-white shadow-lg shadow-purple-500/25'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`
              }
            >
              <span className="flex items-center gap-2 relative z-10">
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"
                  />
                </svg>
                Discover
              </span>
              {isActive && (
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl blur" />
              )}
            </NavLink>

            <NavLink
              to="/favorites"
              className={({ isActive }) =>
                `relative px-5 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ${
                  isActive
                    ? 'bg-gradient-to-r from-purple-600/90 to-pink-600/90 text-white shadow-lg shadow-purple-500/25'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`
              }
            >
              <span className="flex items-center gap-2 relative z-10">
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"
                  />
                </svg>
                Favorites
              </span>
              {isActive && (
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl blur" />
              )}
            </NavLink>
          </div>

          {/* User Menu */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-xl bg-white/3 border border-white/10 hover:border-purple-500/30 transition-all duration-300 group">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-xs font-bold text-white group-hover:scale-110 transition-transform duration-300">
                {user?.username?.[0]?.toUpperCase() || 'U'}
              </div>
              <span className="text-sm font-medium text-gray-200">{user?.username}</span>
              <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" title="Online" />
            </div>
            <button
              onClick={logout}
              className="relative p-2.5 rounded-xl text-gray-400 hover:text-red-400 hover:bg-red-500/10 transition-all duration-300 group"
              title="Logout"
            >
              <svg
                className="w-5 h-5 group-hover:rotate-90 transition-transform duration-300"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9"
                />
              </svg>
              <span className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-xs text-gray-500 bg-gray-900/90 px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                Logout
              </span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}
