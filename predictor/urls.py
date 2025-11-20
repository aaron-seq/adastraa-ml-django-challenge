from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_and_predict, name='upload_predict'),
    path('download/', views.download_predictions, name='download'),
    path('about/', views.about, name='about'),
]
