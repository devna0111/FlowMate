from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os


class UploadedFile(models.Model):
    """사용자가 업로드한 파일 정보"""
    
    FILE_TYPES = [
        ('pdf', 'PDF'),
        ('docx', 'Word Document'),
        ('txt', 'Text File'),
        ('xlsx', 'Excel File'),
        ('pptx', 'PowerPoint'),
        ('other', 'Other'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_files')
    original_filename = models.CharField(max_length=255, verbose_name="원본 파일명")
    file_path = models.CharField(max_length=500, verbose_name="저장 경로")
    file_type = models.CharField(max_length=10, choices=FILE_TYPES, default='other')
    file_size = models.BigIntegerField(verbose_name="파일 크기(bytes)", default=0)
    upload_date = models.DateTimeField(default=timezone.now, verbose_name="업로드 날짜")
    is_processed = models.BooleanField(default=False, verbose_name="벡터화 완료 여부")
    description = models.TextField(blank=True, null=True, verbose_name="파일 설명")
    
    class Meta:
        verbose_name = "업로드된 파일"
        verbose_name_plural = "업로드된 파일들"
        ordering = ['-upload_date']
    
    def __str__(self):
        return f"{self.user.username} - {self.original_filename}"
    
    @property
    def file_size_mb(self):
        """파일 크기를 MB 단위로 반환"""
        return round(self.file_size / (1024 * 1024), 2)
    
    def delete(self, *args, **kwargs):
        """모델 삭제 시 실제 파일도 함께 삭제"""
        if self.file_path and os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
            except OSError:
                pass  # 파일 삭제 실패해도 DB 레코드는 삭제
        super().delete(*args, **kwargs)


class ChatSession(models.Model):
    """사용자의 채팅 세션 관리"""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_sessions')
    session_id = models.CharField(max_length=100, unique=True, verbose_name="세션 ID")
    title = models.CharField(max_length=200, blank=True, null=True, verbose_name="세션 제목")
    created_date = models.DateTimeField(default=timezone.now, verbose_name="생성 날짜")
    last_activity = models.DateTimeField(auto_now=True, verbose_name="마지막 활동")
    is_active = models.BooleanField(default=True, verbose_name="활성 상태")
    
    class Meta:
        verbose_name = "채팅 세션"
        verbose_name_plural = "채팅 세션들"
        ordering = ['-last_activity']
    
    def __str__(self):
        return f"{self.user.username} - {self.title or self.session_id[:20]}"


class ChatMessage(models.Model):
    """채팅 메시지 저장"""
    
    MESSAGE_TYPES = [
        ('user', '사용자'),
        ('assistant', 'AI 어시스턴트'),
        ('system', '시스템'),
    ]
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    content = models.TextField(verbose_name="메시지 내용")
    timestamp = models.DateTimeField(default=timezone.now, verbose_name="전송 시간")
    file_reference = models.ForeignKey(
        UploadedFile, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        verbose_name="참조 파일"
    )
    
    class Meta:
        verbose_name = "채팅 메시지"
        verbose_name_plural = "채팅 메시지들"
        ordering = ['timestamp']
    
    def __str__(self):
        return f"{self.session.user.username} - {self.message_type} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"


class GeneratedReport(models.Model):
    """생성된 보고서/발표자료 관리"""
    
    REPORT_TYPES = [
        ('report', '보고서'),
        ('presentation', '발표자료'),
        ('summary', '요약본'),
    ]
    
    STATUS_CHOICES = [
        ('generating', '생성 중'),
        ('completed', '생성 완료'),
        ('failed', '생성 실패'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='generated_reports')
    title = models.CharField(max_length=200, verbose_name="제목")
    report_type = models.CharField(max_length=15, choices=REPORT_TYPES)
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='generating')
    
    # 원본 파일 정보
    source_file = models.ForeignKey(
        UploadedFile, 
        on_delete=models.CASCADE,
        related_name='generated_reports',
        verbose_name="원본 파일"
    )
    
    # 생성된 파일 정보
    output_filename = models.CharField(max_length=255, verbose_name="출력 파일명")
    output_file_path = models.CharField(max_length=500, verbose_name="출력 파일 경로")
    markdown_content = models.TextField(blank=True, null=True, verbose_name="마크다운 내용")
    
    # 생성 요청 정보
    user_query = models.TextField(verbose_name="사용자 요청")
    generation_date = models.DateTimeField(default=timezone.now, verbose_name="생성 날짜")
    completion_date = models.DateTimeField(null=True, blank=True, verbose_name="완료 날짜")
    
    # 메타데이터
    file_size = models.BigIntegerField(default=0, verbose_name="파일 크기")
    download_count = models.IntegerField(default=0, verbose_name="다운로드 횟수")
    
    class Meta:
        verbose_name = "생성된 보고서"
        verbose_name_plural = "생성된 보고서들"
        ordering = ['-generation_date']
    
    def __str__(self):
        return f"{self.user.username} - {self.title} ({self.get_report_type_display()})"
    
    @property
    def file_size_mb(self):
        """파일 크기를 MB 단위로 반환"""
        return round(self.file_size / (1024 * 1024), 2)
    
    def increment_download_count(self):
        """다운로드 횟수 증가"""
        self.download_count += 1
        self.save(update_fields=['download_count'])
    
    def mark_completed(self):
        """생성 완료 처리"""
        self.status = 'completed'
        self.completion_date = timezone.now()
        self.save(update_fields=['status', 'completion_date'])
    
    def mark_failed(self):
        """생성 실패 처리"""
        self.status = 'failed'
        self.completion_date = timezone.now()
        self.save(update_fields=['status', 'completion_date'])
    
    def delete(self, *args, **kwargs):
        """모델 삭제 시 실제 파일도 함께 삭제"""
        if self.output_file_path and os.path.exists(self.output_file_path):
            try:
                os.remove(self.output_file_path)
            except OSError:
                pass  # 파일 삭제 실패해도 DB 레코드는 삭제
        super().delete(*args, **kwargs)


class UserProfile(models.Model):
    """사용자 프로필 확장"""
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    
    # 사용자 설정
    preferred_language = models.CharField(
        max_length=10,
        choices=[('ko', '한국어'), ('en', 'English')],
        default='ko',
        verbose_name="선호 언어"
    )
    
    # 사용 통계
    total_uploads = models.IntegerField(default=0, verbose_name="총 업로드 수")
    total_reports_generated = models.IntegerField(default=0, verbose_name="총 생성 보고서 수")
    total_chat_messages = models.IntegerField(default=0, verbose_name="총 채팅 메시지 수")
    
    # 계정 정보
    created_date = models.DateTimeField(default=timezone.now, verbose_name="가입 날짜")
    last_login_date = models.DateTimeField(null=True, blank=True, verbose_name="마지막 로그인")
    
    # 저장소 설정
    max_file_size_mb = models.IntegerField(default=50, verbose_name="최대 파일 크기(MB)")
    max_files_count = models.IntegerField(default=100, verbose_name="최대 파일 개수")
    
    class Meta:
        verbose_name = "사용자 프로필"
        verbose_name_plural = "사용자 프로필들"
    
    def __str__(self):
        return f"{self.user.username}의 프로필"
    
    @property
    def used_storage_mb(self):
        """사용 중인 저장공간(MB)"""
        total_size = self.user.uploaded_files.aggregate(
            total=models.Sum('file_size')
        )['total'] or 0
        return round(total_size / (1024 * 1024), 2)
    
    @property
    def storage_usage_percent(self):
        """저장공간 사용률(%)"""
        max_size_bytes = self.max_file_size_mb * self.max_files_count * 1024 * 1024
        used_size_bytes = self.user.uploaded_files.aggregate(
            total=models.Sum('file_size')
        )['total'] or 0
        
        if max_size_bytes == 0:
            return 0
        return round((used_size_bytes / max_size_bytes) * 100, 1)


# User 모델과 UserProfile 자동 연결을 위한 시그널
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """새 사용자 생성 시 자동으로 프로필 생성"""
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """사용자 저장 시 프로필도 함께 저장"""
    if hasattr(instance, 'profile'):
        instance.profile.save()
