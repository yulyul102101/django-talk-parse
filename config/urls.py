from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

from parseaudio.views import AudioAnalysisView, DownloadSegmentsView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', AudioAnalysisView.as_view(), name='audio-parse'),
    path('download/', DownloadSegmentsView.as_view(), name='audio-download'),
    # path('api/', include('parseaudio.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
