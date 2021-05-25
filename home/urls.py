from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [

    
    path('',views.index,name='home'),
    path('dashboard/',views.dashboard,name='dashboard'),
    path('neural/',views.neural,name='neural')
]
