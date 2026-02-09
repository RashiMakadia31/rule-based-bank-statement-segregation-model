from django.urls import path
from .views import PredictAPIView, AutoLearnAPIView, ApproveAPIView

urlpatterns = [
    path('predict/', PredictAPIView.as_view(), name='predict'),
    path('auto-learn/', AutoLearnAPIView.as_view(), name='auto-learn'),
    path('approve/', ApproveAPIView.as_view(), name='approve'),
]