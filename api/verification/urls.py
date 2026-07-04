from django.urls import path
from . import views

urlpatterns = [
    path('', views.serve_frontend, name='home'),
    path('verify/', views.verify_media, name='verify_media'),
    path('verify/history', views.verification_history, name='verification_history'),
    path('verify/history/', views.clear_verification_history, name='clear_verification_history'),
]
