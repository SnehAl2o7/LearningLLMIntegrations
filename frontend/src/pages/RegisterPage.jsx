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
  const [showPassword, setShowPassword] = useState(false);
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
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl p-4 text-sm animate-slideUp flex items-center gap-3">
          <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
          {error}
        </div>
      );
    }
    
    return (
      <div className="bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl p-4 text-sm animate-slideUp space-y-2">
        {Object.entries(error).map(([field, msg]) => (
          <div key={field} className="flex items-center gap-2">
            <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <span className="font-medium capitalize">{field}</span>: {Array.isArray(msg) ? msg.join(', ') : msg}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4 page-enter hero-pattern">
      <div className="glass-card-elevated w-full max-w-md p-8 sm:p-10 rounded-3xl border border-white/10 relative overflow-hidden glow-pulse">
        {/* Animated background elements */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-48 bg-gradient-to-b from-purple-500/10 via-transparent to-pink-500/10 blur-[100px] pointer-events-none" />
        <div className="absolute -top-20 -right-20 w-72 h-72 bg-purple-500/10 rounded-full blur-[150px] animate-float" />
        <div className="absolute -bottom-20 -left-20 w-72 h-72 bg-pink-500/10 rounded-full blur-[150px] animate-float" style={{ animationDelay: '2s' }} />
        
        <div className="text-center mb-10 relative z-10">
          <div className="relative inline-block mb-6">
            <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center shadow-2xl shadow-purple-500/25 animate-pulse-slow">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            </div>
            <div className="absolute -top-2 -right-2 w-4 h-4 bg-amber-400 rounded-full animate-ping opacity-75" />
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold mb-3 gradient-text tracking-tight">
            Create Account
          </h1>
          <p className="text-lg text-gray-300">Join BookLens and start discovering amazing books</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6 relative z-10">
          {renderError()}
          
          {success && (
            <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-xl p-4 text-sm text-center animate-slideUp flex items-center justify-center gap-3">
              <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Account created successfully! Redirecting...
            </div>
          )}

          <div className="space-y-2">
            <label htmlFor="register-username" className="block text-sm font-medium text-gray-300">
              Username
            </label>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 via-transparent to-pink-500/5 rounded-xl opacity-0 peer-focus:opacity-100 transition-opacity duration-300 pointer-events-none" />
              <input
                id="register-username"
                name="username"
                type="text"
                value={formData.username}
                onChange={handleChange}
                className="input-styled w-full bg-white/3 border border-white/10 rounded-xl px-5 py-4 text-white placeholder-gray-500 peer"
                placeholder="Choose a username"
                required
                autoComplete="username"
              />
              <div className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <label htmlFor="register-email" className="block text-sm font-medium text-gray-300">
              Email
            </label>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 via-transparent to-pink-500/5 rounded-xl opacity-0 peer-focus:opacity-100 transition-opacity duration-300 pointer-events-none" />
              <input
                id="register-email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                className="input-styled w-full bg-white/3 border border-white/10 rounded-xl px-5 py-4 text-white placeholder-gray-500 peer"
                placeholder="Enter your email"
                required
                autoComplete="email"
              />
              <div className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <label htmlFor="register-password" className="block text-sm font-medium text-gray-300">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 via-transparent to-pink-500/5 rounded-xl opacity-0 peer-focus:opacity-100 transition-opacity duration-300 pointer-events-none" />
              <input
                id="register-password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                value={formData.password}
                onChange={handleChange}
                className="input-styled w-full bg-white/3 border border-white/10 rounded-xl px-5 py-4 pr-14 text-white placeholder-gray-500 peer"
                placeholder="Create a password"
                required
                autoComplete="new-password"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white transition-colors"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                )}
              </button>
            </div>
          </div>

          <div className="space-y-2">
            <label htmlFor="register-password2" className="block text-sm font-medium text-gray-300">
              Confirm Password
            </label>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 via-transparent to-pink-500/5 rounded-xl opacity-0 peer-focus:opacity-100 transition-opacity duration-300 pointer-events-none" />
              <input
                id="register-password2"
                name="password2"
                type={showPassword ? 'text' : 'password'}
                value={formData.password2}
                onChange={handleChange}
                className="input-styled w-full bg-white/3 border border-white/10 rounded-xl px-5 py-4 pr-14 text-white placeholder-gray-500 peer"
                placeholder="Confirm your password"
                required
                autoComplete="new-password"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white transition-colors"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                )}
              </button>
            </div>
          </div>

          <button
            id="register-submit"
            type="submit"
            disabled={isLoading || success}
            className="btn-primary w-full rounded-xl py-4 font-semibold text-white mt-2 flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden"
          >
            {isLoading ? (
              <>
                <div className="flex gap-1 items-center">
                  <div className="w-2 h-2 bg-white rounded-full loading-dot"></div>
                  <div className="w-2 h-2 bg-white rounded-full loading-dot"></div>
                  <div className="w-2 h-2 bg-white rounded-full loading-dot"></div>
                </div>
                <span>Creating account...</span>
              </>
            ) : (
              <>
                <span>Create Account</span>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                </svg>
              </>
            )}
            <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 opacity-0 hover:opacity-100 transition-opacity duration-300" />
          </button>
        </form>

        <p className="mt-8 text-center text-gray-400 relative z-10 text-sm">
          Already have an account?{' '}
          <Link to="/login" className="text-purple-400 hover:text-purple-300 font-medium transition-colors relative">
            Sign in
            <span className="absolute bottom-0 left-0 w-full h-0.5 bg-purple-400 scale-x-0 hover:scale-x-100 transition-transform duration-300 origin-left" />
          </Link>
        </p>
      </div>
    </div>
  );
}
