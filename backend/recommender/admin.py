from django.contrib import admin
from .models import FavoriteBook


@admin.register(FavoriteBook)
class FavoriteBookAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'isbn13', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('user__username', 'isbn13')
    ordering = ('-created_at',)