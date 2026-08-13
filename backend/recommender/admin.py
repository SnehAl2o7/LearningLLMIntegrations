from django.contrib import admin
from .models import FavoriteBook


@admin.register(FavoriteBook)
class FavoriteBookAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'isbn13', 'title', 'authors', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__username', 'isbn13', 'title', 'authors')
    ordering = ('-created_at',)