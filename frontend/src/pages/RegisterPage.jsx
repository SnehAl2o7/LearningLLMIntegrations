import React, { useState } from 'react';
import { Link, useNavigate, Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function RegisterPage() {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    password2: ''
  });
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { register, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  if (isAuthenticated) {
    return <Navigate to="/discover" replace />;
  }

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    
    if (formData.password !== formData.password2) {
      setError('Passwords do not match.');
      return;
    }

    setIsLoading(true);
    
    try {
      await register(formData.username, formData.email, formData.password, formData.password2);
      setSuccess(true);
      setTimeout(() => {
        navigate('/login');
      }, 1500);
    } catch (err) {
      const data = err.response?.data;
      if (data) {
        if (typeof data === 'string') {
          setError(data);
        } else if (data.detail) {
          setError(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
        } else {
          // Field-level errors object from Django
          setError(data);
        }
      } else {
        setError('Network error. Please check your connection.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const renderError = () => {
    if (!error) return null;
    
    if (typeof error === 'string') {
      return (
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg p-3 text-sm">
          {error}
        </div>
      );
    }
    
    return (
      <div className="bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg p-3 text-sm space-y-1">
        {Object.entries(error).map(([field, msg]) => (
          <div key={field}>
            <span className="font-semibold capitalize">{field}</span>: {msg}
          </div>
        ))}
      </div>
    );
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
          <p className="text-gray-400">Create an account to get started</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 relative z-10">
          {renderError()}
          
          {success && (
            <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-lg p-3 text-sm text-center">
              Account created successfully! Redirecting...
            </div>
          )}

          <div className="space-y-1">
            <label className="block text-sm text-gray-400 font-medium">Username</label>
            <input
              id="register-username"
              name="username"
              type="text"
              value={formData.username}
              onChange={handleChange}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:border-purple-500 input-glow text-white placeholder-gray-500 outline-none transition-all"
              placeholder="Choose a username"
              required
            />
          </div>

          <div className="space-y-1">
            <label className="block text-sm text-gray-400 font-medium">Email</label>
            <input
              id="register-email"
              name="email"
              type="email"
              value={formData.email}
              onChange={handleChange}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:border-purple-500 input-glow text-white placeholder-gray-500 outline-none transition-all"
              placeholder="Enter your email"
              required
            />
          </div>

          <div className="space-y-1">
            <label className="block text-sm text-gray-400 font-medium">Password</label>
            <input
              id="register-password"
              name="password"
              type="password"
              value={formData.password}
              onChange={handleChange}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:border-purple-500 input-glow text-white placeholder-gray-500 outline-none transition-all"
              placeholder="Create a password"
              required
            />
          </div>

          <div className="space-y-1">
            <label className="block text-sm text-gray-400 font-medium">Confirm Password</label>
            <input
              id="register-password2"
              name="password2"
              type="password"
              value={formData.password2}
              onChange={handleChange}
              className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:border-purple-500 input-glow text-white placeholder-gray-500 outline-none transition-all"
              placeholder="Confirm your password"
              required
            />
          </div>

          <button
            id="register-submit"
            type="submit"
            disabled={isLoading || success}
            className="btn-primary w-full rounded-xl py-3 font-semibold text-white mt-6 flex items-center justify-center transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <div className="flex space-x-1">
                <div className="loading-dot w-2 h-2 bg-white rounded-full animate-bounce"></div>
                <div className="loading-dot w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="loading-dot w-2 h-2 bg-white rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            ) : (
              'Create Account'
            )}
          </button>
        </form>

        <p className="mt-6 text-center text-gray-400 relative z-10 text-sm">
          Already have an account?{' '}
          <Link to="/login" className="text-purple-400 hover:text-purple-300 font-medium transition-colors">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
