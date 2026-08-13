import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex gap-2">
          <div className="w-3 h-3 rounded-full bg-purple-500 loading-dot" />
          <div className="w-3 h-3 rounded-full bg-purple-500 loading-dot" />
          <div className="w-3 h-3 rounded-full bg-purple-500 loading-dot" />
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return children;
}
