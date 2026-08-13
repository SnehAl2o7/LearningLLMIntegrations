from django.db import models
from django.contrib.auth.models import User


class FavoriteBook(models.Model):
    """
    Represents a book favorited by a user.
    Enforces a unique constraint on (user, isbn13) so a user
    cannot favorite the same book twice.
    """
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='favorite_books',
    )
    isbn13 = models.CharField(max_length=13)
    title = models.CharField(max_length=500, blank=True, default='')
    authors = models.CharField(max_length=500, blank=True, default='')
    thumbnail = models.URLField(max_length=1000, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'isbn13')
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.isbn13}"