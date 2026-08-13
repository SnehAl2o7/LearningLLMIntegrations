import React, { useState } from 'react';
import { Link, useNavigate, Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  if (isAuthenticated) {
    return <Navigate to="/discover" replace />;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);
    
    try {
      await login(username, password);
      navigate('/discover');
    } catch (err) {
      const data = err.response?.data;
      if (data) {
        // Django validation errors can be { detail: '...' } or { field: ['error'] }
        if (typeof data === 'string') {
          setError(data);
        } else if (data.detail) {
          // Handle nested detail arrays from DRF
          const detail = Array.isArray(data.detail) ? data.detail[0] : data.detail;
          setError(typeof detail === 'object' ? detail.string || JSON.stringify(detail) : detail);
        } else {
          // Field-level errors: { username: ['...'], password: ['...'] }
          const msgs = Object.entries(data)
            .map(([key, val]) => `${key}: ${Array.isArray(val) ? val.join(', ') : val}`)
            .join('. ');
          setError(msgs || 'Invalid credentials.');
        }
      } else {
        setError('Network error. Please check your connection.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4 page-enter">
      <div className="glass-card w-full max-w-md p-8 rounded-2xl border border-white/10 relative overflow-hidden">
        {/* Glow effect */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-32 bg-purple-500/20 blur-[100px] pointer-events-none" />
        
        <div className="text-center mb-8 relative z-10">
          <div className="text-5xl mb-4">📚</div>
          <h1 className="text-4xl font-bold mb-2">
            <span className="gradient-text">BookLens</span>
          </h1>
          <p className="text-gray-400">Sign in to discover your next favorite book</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5 relative z-10">
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg p-3 text-sm">
              {error}
            </div>
          )}

          <div className="space-y-1">
            <label className="block text-sm text-gray-400 font-medium">Username</label>
            <input
              id="login-username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:border-purple-500 input-glow text-white placeholder-gray-500 outline-none transition-all"
              placeholder="Enter your username"
              required
            />
          </div>

          <div className="space-y-1">
            <label className="block text-sm text-gray-400 font-medium">Password</label>
            <input
              id="login-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:border-purple-500 input-glow text-white placeholder-gray-500 outline-none transition-all"
              placeholder="Enter your password"
              required
            />
          </div>

          <button
            id="login-submit"
            type="submit"
            disabled={isLoading}
            className="btn-primary w-full rounded-xl py-3 font-semibold text-white mt-4 flex items-center justify-center transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <div className="flex space-x-1">
                <div className="loading-dot w-2 h-2 bg-white rounded-full animate-bounce"></div>
                <div className="loading-dot w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="loading-dot w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <p className="mt-6 text-center text-gray-400 relative z-10 text-sm">
          Don't have an account?{' '}
          <Link to="/register" className="text-purple-400 hover:text-purple-300 font-medium transition-colors">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
