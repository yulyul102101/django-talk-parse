import io
import os
import zipfile
import subprocess

from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views import View
from django.shortcuts import render
from django.utils import timezone
from .models import OriginalAudio, SegmentAudio
from .forms import AudioUploadForm
from .parsing.service import parse_audio, get_sorted_segments, detect_audio, bynary_to_content_file
from .utils import get_installed_ollama_models, stop_running_model

class AudioAnalysisView(View):
    template_name = "index.html"


    def get(self, request):
        form = AudioUploadForm()
        models = get_installed_ollama_models()
        return render(request, self.template_name, {"form": form, "ollama_models": models})

    def post(self, request):
        form = AudioUploadForm(request.POST, request.FILES)
        selected_model = request.POST.get("ollama_model")  # 사용자가 선택한 모델

        if form.is_valid():
            audio_file = form.cleaned_data['audio_file']

            original_audio = OriginalAudio.objects.create(
                audio_file=audio_file,
                create_date=timezone.now(),
            )

            audio_path = original_audio.audio_file.path

            try:
                transcripts = parse_audio(audio_path)
                segments = get_sorted_segments(transcripts)
                saved_segments = []

                for idx, segment in enumerate(segments, start=1):
                    new_segment = SegmentAudio(
                        original_id=original_audio,
                        segment_index=idx,
                        speaker=segment['speaker'],
                        transcription=segment['text'],
                        start_time=segment['start'],
                        end_time=segment['end']
                    )
                    new_segment_file = bynary_to_content_file(segment['audio'], idx)
                    new_segment.audio_file.save(f"segment_{idx}.wav", new_segment_file)
                    saved_segments.append(new_segment)

                stop_running_model(model_name=selected_model)
                detect_res = detect_audio(transcripts, model_name=selected_model)

                original_audio.is_phishing = detect_res["judgment"]
                original_audio.phishing_reason = detect_res["evidence"]
                original_audio.save()

                models = get_installed_ollama_models()
                context = {
                    "original_audio": original_audio,
                    "segments": saved_segments,
                    "ollama_models": models,
                }
                return render(request, self.template_name, context)

            except ValueError as ve:
                return JsonResponse({"error": str(ve)}, status=400)
            except Exception as e:
                return JsonResponse({"error": str(e)}, status=500)

        models = get_installed_ollama_models()
        return render(request, self.template_name, {"form": form, "ollama_models": models})


class DownloadSegmentsView(View):
    def post(self, request, *args, **kwargs):
        # 다운로드 타입 (text, wav)
        download_types = request.POST.getlist('download_type')  # ['text', 'wav']
        wav_mode = request.POST.get('wav_mode', 'individual')  # 'individual' or 'merged'

        # 선택된 세그먼트 PK
        selected_segments_pks = request.POST.getlist('segments')
        selected_segments_pks = [int(pk) for pk in selected_segments_pks]

        # 세그먼트 객체 가져오기
        segments = SegmentAudio.objects.filter(pk__in=selected_segments_pks)

        # Zip 파일 생성
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:

            # 텍스트 파일 생성 (선택한 경우)
            if 'text' in download_types:
                transcript_content = ""
                for segment in segments:
                    transcript_content += (
                        f"[Speaker: {segment.speaker}]\n"
                        f"Start: {segment.start_time}s\n"
                        f"End: {segment.end_time}s\n"
                        f"Text: {segment.transcription}\n\n"
                    )

                zf.writestr("transcripts.txt", transcript_content)

            # WAV 파일 생성 (선택한 경우)
            if 'wav' in download_types:
                if wav_mode == 'individual':
                    for segment in segments:
                        if segment.audio_file:
                            file_path = segment.audio_file.path
                            zf.write(file_path, arcname=f"segment_{segment.segment_index}.wav")
                elif wav_mode == 'merged':
                    merged_audio_path = self.merge_audio_files(segments)
                    zf.write(merged_audio_path, arcname="merged_audio.wav")
                    os.remove(merged_audio_path)  # 임시 파일 삭제

        # Zip 파일 반환
        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type="application/zip")
        response["Content-Disposition"] = "attachment; filename=segments.zip"
        return response

    def merge_audio_files(self, segments):
        """선택된 세그먼트의 오디오 파일을 병합하여 단일 WAV 파일 생성."""
        import soundfile as sf
        import numpy as np

        merged_audio = []
        sample_rate = None

        for segment in segments:
            if segment.audio_file:
                data, sr = sf.read(segment.audio_file.path)
                merged_audio.append(data)
                if sample_rate is None:
                    sample_rate = sr

        merged_audio = np.concatenate(merged_audio)

        # 병합된 오디오 파일 저장 경로
        merged_audio_path = os.path.join(settings.MEDIA_ROOT, "merged_audio.wav")
        sf.write(merged_audio_path, merged_audio, sample_rate)

        return merged_audio_path