"""
Tests for the recommender app.

Covers:
- User registration
- User login (JWT)
- Favorites CRUD operations
- Recommendations endpoint
"""
from django.contrib.auth.models import User
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient
from rest_framework_simplejwt.tokens import RefreshToken

from .models import FavoriteBook


class AuthTests(TestCase):
    """Test authentication endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.register_url = '/api/auth/register/'
        self.login_url = '/api/auth/login/'
        self.valid_user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'TestPass123!',
            'password2': 'TestPass123!',
        }

    def test_register_success(self):
        """Test successful user registration."""
        response = self.client.post(self.register_url, self.valid_user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('id', response.data)
        self.assertEqual(response.data['username'], 'testuser')
        self.assertEqual(response.data['email'], 'test@example.com')
        self.assertTrue(User.objects.filter(username='testuser').exists())

    def test_register_password_mismatch(self):
        """Test registration fails with mismatched passwords."""
        data = self.valid_user_data.copy()
        data['password2'] = 'DifferentPass123!'
        response = self.client.post(self.register_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('password', response.data)

    def test_register_duplicate_username(self):
        """Test registration fails with duplicate username."""
        User.objects.create_user(username='testuser', email='existing@example.com', password='TestPass123!')
        response = self.client.post(self.register_url, self.valid_user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_login_success(self):
        """Test successful login returns JWT tokens."""
        User.objects.create_user(username='testuser', email='test@example.com', password='TestPass123!')
        response = self.client.post(self.login_url, {
            'username': 'testuser',
            'password': 'TestPass123!',
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)
        self.assertIn('user', response.data)

    def test_login_invalid_credentials(self):
        """Test login fails with invalid credentials."""
        response = self.client.post(self.login_url, {
            'username': 'testuser',
            'password': 'WrongPass123!',
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('detail', response.data)

    def test_login_inactive_user(self):
        """Test login fails for inactive user."""
        user = User.objects.create_user(username='inactive', email='inactive@example.com', password='TestPass123!')
        user.is_active = False
        user.save()
        response = self.client.post(self.login_url, {
            'username': 'inactive',
            'password': 'TestPass123!',
        }, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class FavoriteBookTests(TestCase):
    """Test favorites CRUD endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', email='test@example.com', password='TestPass123!')
        self.other_user = User.objects.create_user(username='otheruser', email='other@example.com', password='TestPass123!')
        self.isbn13 = '9780002005883'
        self.isbn13_2 = '9780002261982'

        # Get JWT token for authentication
        refresh = RefreshToken.for_user(self.user)
        self.access_token = str(refresh.access_token)
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {self.access_token}')

        self.list_url = '/api/favorites/'
        self.add_url = '/api/favorites/add/'
        self.remove_url = f'/api/favorites/remove/{self.isbn13}/'

    def test_list_favorites_empty(self):
        """Test listing favorites when none exist."""
        response = self.client.get(self.list_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], 0)
        self.assertEqual(response.data['favorites'], [])

    def test_add_favorite_success(self):
        """Test adding a book to favorites."""
        response = self.client.post(self.add_url, {'isbn13': self.isbn13}, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('favorite', response.data)
        self.assertEqual(response.data['favorite']['isbn13'], self.isbn13)
        self.assertTrue(FavoriteBook.objects.filter(user=self.user, isbn13=self.isbn13).exists())

    def test_add_favorite_invalid_isbn(self):
        """Test adding favorite with invalid ISBN format."""
        response = self.client.post(self.add_url, {'isbn13': 'invalid'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_add_favorite_duplicate(self):
        """Test adding the same book twice returns conflict."""
        FavoriteBook.objects.create(user=self.user, isbn13=self.isbn13)
        response = self.client.post(self.add_url, {'isbn13': self.isbn13}, format='json')
        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)

    def test_list_favorites_with_data(self):
        """Test listing favorites returns user's books."""
        FavoriteBook.objects.create(user=self.user, isbn13=self.isbn13)
        FavoriteBook.objects.create(user=self.user, isbn13=self.isbn13_2)
        # Other user's favorite should not appear
        FavoriteBook.objects.create(user=self.other_user, isbn13=self.isbn13)

        response = self.client.get(self.list_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], 2)
        isbns = [f['isbn13'] for f in response.data['favorites']]
        self.assertIn(self.isbn13, isbns)
        self.assertIn(self.isbn13_2, isbns)

    def test_remove_favorite_success(self):
        """Test removing a book from favorites."""
        FavoriteBook.objects.create(user=self.user, isbn13=self.isbn13)
        response = self.client.delete(self.remove_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(FavoriteBook.objects.filter(user=self.user, isbn13=self.isbn13).exists())

    def test_remove_favorite_not_found(self):
        """Test removing non-existent favorite returns 404."""
        response = self.client.delete(self.remove_url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_remove_favorite_other_user(self):
        """Test user cannot remove another user's favorite."""
        FavoriteBook.objects.create(user=self.other_user, isbn13=self.isbn13)
        response = self.client.delete(self.remove_url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_favorites_requires_authentication(self):
        """Test favorites endpoints require authentication."""
        unauthenticated_client = APIClient()
        response = unauthenticated_client.get(self.list_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        response = unauthenticated_client.post(self.add_url, {'isbn13': self.isbn13}, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

        response = unauthenticated_client.delete(self.remove_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class RecommendationTests(TestCase):
    """Test recommendations endpoint."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', email='test@example.com', password='TestPass123!')
        refresh = RefreshToken.for_user(self.user)
        self.access_token = str(refresh.access_token)
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {self.access_token}')
        self.recommend_url = '/api/recommendations/'

    def test_recommendations_requires_authentication(self):
        """Test recommendations endpoint requires authentication."""
        unauthenticated_client = APIClient()
        response = unauthenticated_client.post(self.recommend_url, {'query': 'test'}, format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_recommendations_invalid_request(self):
        """Test recommendations with missing query returns 400."""
        response = self.client.post(self.recommend_url, {}, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('query', response.data)

    def test_recommendations_valid_request_structure(self):
        """Test recommendations endpoint accepts valid request structure."""
        # This tests the serializer validation, not the actual recommendation logic
        # which requires NVIDIA_API_KEY and Chroma DB
        response = self.client.post(self.recommend_url, {
            'query': 'A book about nature',
            'category': 'All',
            'tone': 'All',
            'top_k': 10,
        }, format='json')
        # Will return 503 if NVIDIA_API_KEY not set, but should not be 400
        self.assertNotEqual(response.status_code, status.HTTP_400_BAD_REQUEST)