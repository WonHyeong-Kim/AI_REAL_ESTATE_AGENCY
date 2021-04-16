"""ai_real_estate_agency URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls.conf import include

from predictapp.views import views2, views1

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views1.MainFunc),
#     path('info', views.InfoFunc),
#     path('predict/', include('predictapp.urls')),
    path('predict/featurePrice', views2.FeaturePriceFunc),
    path('predict/', views2.PredictFunc),
    path('predict/info', views1.InfoFunc),
    path('model/', views1.ModelFunc),
    path('chart/', views1.ChartFunc),
    path('predict_price/', views2.predict_price),
    path('predict_price/predict', views2.predict_modeling),
    path('chart/gu_chart', views1.GuChart),
    path('loading/', views1.LoadingFunc),
]
# cd C:\work\psou\ai_real_estate_agency
# python manage.py createsuperuser