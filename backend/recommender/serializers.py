from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken

from .models import FavoriteBook


class RegisterSerializer(serializers.ModelSerializer):
    """
    Serializer for user registration.
    Validates password strength and creates a new User.
    """
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password],
        style={'input_type': 'password'},
    )
    password2 = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'},
    )

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'password2')

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({
                'password': 'Password fields did not match.'
            })
        return attrs

    def create(self, validated_data):
        validated_data.pop('password2')
        user = User.objects.create_user(**validated_data)
        return user


class LoginSerializer(serializers.Serializer):
    """
    Serializer for JWT login.
    Validates credentials and returns access + refresh tokens.
    """
    username = serializers.CharField(required=True)
    password = serializers.CharField(
        required=True,
        write_only=True,
        style={'input_type': 'password'},
    )

    def validate(self, attrs):
        username = attrs.get('username')
        password = attrs.get('password')

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            raise serializers.ValidationError({
                'detail': 'Invalid credentials.'
            })

        if not user.check_password(password):
            raise serializers.ValidationError({
                'detail': 'Invalid credentials.'
            })

        if not user.is_active:
            raise serializers.ValidationError({
                'detail': 'User account is disabled.'
            })

        refresh = RefreshToken.for_user(user)

        return {
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
            },
            'access': str(refresh.access_token),
            'refresh': str(refresh),
        }


class FavoriteBookSerializer(serializers.ModelSerializer):
    """
    Serializer for the FavoriteBook model.
    """
    class Meta:
        model = FavoriteBook
        fields = ('id', 'isbn13', 'title', 'authors', 'thumbnail', 'created_at')
        read_only_fields = ('id', 'created_at')


class FavoriteAddSerializer(serializers.Serializer):
    """
    Serializer for adding a book to favorites.
    """
    isbn13 = serializers.CharField(
        required=True,
        max_length=13,
        min_length=13,
        help_text='The 13-digit ISBN of the book to favorite.',
    )
    title = serializers.CharField(
        required=False,
        default='',
        max_length=500,
        allow_blank=True,
    )
    authors = serializers.CharField(
        required=False,
        default='',
        max_length=500,
        allow_blank=True,
    )
    thumbnail = serializers.CharField(
        required=False,
        default='',
        max_length=1000,
        allow_blank=True,
    )

    def validate_isbn13(self, value):
        if not value.isdigit():
            raise serializers.ValidationError('ISBN must contain only digits.')
        return value


class RecommendationRequestSerializer(serializers.Serializer):
    """
    Serializer for the POST /api/recommendations/ endpoint.
    """
    query = serializers.CharField(
        required=True,
        max_length=1000,
        help_text='Natural-language description of the book you want.',
    )
    category = serializers.CharField(
        required=False,
        default='All',
        max_length=100,
        help_text='Optional category filter (e.g. Fiction, Nonfiction).',
    )
    tone = serializers.CharField(
        required=False,
        default='All',
        max_length=50,
        help_text='Optional emotional tone (Happy, Surprising, Angry, Suspenseful, Sad).',
    )
    top_k = serializers.IntegerField(
        required=False,
        default=16,
        min_value=1,
        max_value=50,
        help_text='Number of recommendations to return.',
    )


class BookRecommendationSerializer(serializers.Serializer):
    """
    Serializer for a single book recommendation response.
    """
    isbn13 = serializers.CharField()
    isbn10 = serializers.CharField(required=False, allow_blank=True)
    title = serializers.CharField(required=False, allow_blank=True)
    authors = serializers.CharField(required=False, allow_blank=True)
    categories = serializers.CharField(required=False, allow_blank=True)
    simple_categories = serializers.CharField(required=False, allow_blank=True)
    thumbnail = serializers.CharField(required=False, allow_blank=True)
    large_thumbnail = serializers.CharField(required=False, allow_blank=True)
    description = serializers.CharField(required=False, allow_blank=True)
    published_year = serializers.FloatField(required=False, allow_null=True)
    average_rating = serializers.FloatField(required=False, allow_null=True)
    num_pages = serializers.FloatField(required=False, allow_null=True)
    ratings_count = serializers.FloatField(required=False, allow_null=True)
    title_and_subtitle = serializers.CharField(required=False, allow_blank=True)