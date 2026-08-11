from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('auth/register/', views.RegisterView.as_view(), name='register'),
    path('auth/login/', views.LoginView.as_view(), name='login'),

    # Recommendations
    path('recommendations/', views.RecommendationView.as_view(), name='recommendations'),

    # Favorites
    path('favorites/', views.FavoriteListView.as_view(), name='favorites-list'),
    path('favorites/add/', views.FavoriteAddView.as_view(), name='favorites-add'),
    path('favorites/remove/<str:isbn13>/', views.FavoriteRemoveView.as_view(), name='favorites-remove'),
]