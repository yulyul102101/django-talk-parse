from django import forms
from .models import OriginalAudio


class AudioUploadForm(forms.ModelForm):
    class Meta:
        model = OriginalAudio
        fields = ['audio_file']