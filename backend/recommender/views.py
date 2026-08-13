from django.db import IntegrityError
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import FavoriteBook
from .serializers import (
    RegisterSerializer,
    LoginSerializer,
    FavoriteBookSerializer,
    FavoriteAddSerializer,
    RecommendationRequestSerializer,
    BookRecommendationSerializer,
)
from .services import get_recommendations


class RegisterView(APIView):
    """
    POST /api/auth/register/
    Register a new user.
    """
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response(
                {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'message': 'User registered successfully.',
                },
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    """
    POST /api/auth/login/
    Authenticate a user and return JWT access + refresh tokens.
    """
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            return Response(serializer.validated_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class RecommendationView(APIView):
    """
    POST /api/recommendations/
    Returns book recommendations from the Chroma vector store.
    Protected: requires a valid JWT access token.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = RecommendationRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        query = serializer.validated_data['query']
        category = serializer.validated_data.get('category', 'All')
        tone = serializer.validated_data.get('tone', 'All')
        top_k = serializer.validated_data.get('top_k', 16)

        try:
            recommendations = get_recommendations(
                query=query,
                category=category,
                tone=tone,
                final_top_k=top_k,
            )
        except RuntimeError as e:
            return Response(
                {'detail': str(e)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except Exception as e:
            return Response(
                {'detail': f'Recommendation service error: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        response_serializer = BookRecommendationSerializer(
            recommendations,
            many=True,
        )
        return Response(
            {
                'query': query,
                'category': category,
                'tone': tone,
                'count': len(recommendations),
                'recommendations': response_serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class FavoriteListView(APIView):
    """
    GET /api/favorites/
    Returns all favorited ISBNs for the authenticated user.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        favorites = FavoriteBook.objects.filter(user=request.user)
        serializer = FavoriteBookSerializer(favorites, many=True)
        return Response(
            {
                'count': favorites.count(),
                'favorites': serializer.data,
            },
            status=status.HTTP_200_OK,
        )


class FavoriteAddView(APIView):
    """
    POST /api/favorites/add/
    Adds an ISBN to the authenticated user's favorites.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = FavoriteAddSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        isbn13 = serializer.validated_data['isbn13']
        title = serializer.validated_data.get('title', '')
        authors = serializer.validated_data.get('authors', '')
        thumbnail = serializer.validated_data.get('thumbnail', '')

        # Check if already favorited
        if FavoriteBook.objects.filter(user=request.user, isbn13=isbn13).exists():
            return Response(
                {'detail': 'This book is already in your favorites.'},
                status=status.HTTP_409_CONFLICT,
            )

        try:
            favorite = FavoriteBook.objects.create(
                user=request.user,
                isbn13=isbn13,
                title=title,
                authors=authors,
                thumbnail=thumbnail,
            )
        except IntegrityError:
            return Response(
                {'detail': 'This book is already in your favorites.'},
                status=status.HTTP_409_CONFLICT,
            )

        response_serializer = FavoriteBookSerializer(favorite)
        return Response(
            {
                'message': 'Book added to favorites.',
                'favorite': response_serializer.data,
            },
            status=status.HTTP_201_CREATED,
        )


class FavoriteRemoveView(APIView):
    """
    DELETE /api/favorites/remove/<str:isbn13>/
    Deletes an ISBN from the authenticated user's favorites.
    """
    permission_classes = [IsAuthenticated]

    def delete(self, request, isbn13):
        try:
            favorite = FavoriteBook.objects.get(
                user=request.user,
                isbn13=isbn13,
            )
        except FavoriteBook.DoesNotExist:
            return Response(
                {'detail': 'Favorite not found.'},
                status=status.HTTP_404_NOT_FOUND,
            )

        favorite.delete()
        return Response(
            {'message': 'Book removed from favorites.'},
            status=status.HTTP_200_OK,
        )