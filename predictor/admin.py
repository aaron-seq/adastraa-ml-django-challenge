from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('id', 'created_at', 'num_predictions')
    list_filter = ('created_at',)
    ordering = ('-created_at',)
