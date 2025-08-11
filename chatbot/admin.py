from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import UploadedFile, ChatSession, ChatMessage, GeneratedReport, UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = [
        'user', 'preferred_language', 'total_uploads', 
        'total_reports_generated', 'used_storage_mb', 'created_date'
    ]
    list_filter = ['preferred_language', 'created_date']
    search_fields = ['user__username', 'user__email']
    readonly_fields = [
        'total_uploads', 'total_reports_generated', 'total_chat_messages',
        'created_date', 'used_storage_mb', 'storage_usage_percent'
    ]
    
    fieldsets = (
        ('사용자 정보', {
            'fields': ('user', 'preferred_language')
        }),
        ('사용 통계', {
            'fields': (
                'total_uploads', 'total_reports_generated', 'total_chat_messages',
                'used_storage_mb', 'storage_usage_percent'
            )
        }),
        ('저장소 설정', {
            'fields': ('max_file_size_mb', 'max_files_count')
        }),
        ('계정 정보', {
            'fields': ('created_date', 'last_login_date')
        }),
    )


@admin.register(UploadedFile)
class UploadedFileAdmin(admin.ModelAdmin):
    list_display = [
        'original_filename', 'user', 'file_type', 'file_size_mb', 
        'is_processed', 'upload_date'
    ]
    list_filter = ['file_type', 'is_processed', 'upload_date', 'user']
    search_fields = ['original_filename', 'user__username', 'description']
    readonly_fields = ['upload_date', 'file_size_mb']
    
    fieldsets = (
        ('파일 정보', {
            'fields': ('original_filename', 'file_path', 'file_type', 'file_size', 'file_size_mb')
        }),
        ('사용자 정보', {
            'fields': ('user', 'upload_date')
        }),
        ('처리 상태', {
            'fields': ('is_processed', 'description')
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['title_display', 'user', 'message_count', 'created_date', 'last_activity', 'is_active']
    list_filter = ['is_active', 'created_date', 'last_activity']
    search_fields = ['title', 'user__username', 'session_id']
    readonly_fields = ['session_id', 'created_date', 'message_count']
    
    def title_display(self, obj):
        return obj.title or f"세션 {obj.session_id[:8]}..."
    title_display.short_description = "세션 제목"
    
    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = "메시지 수"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user').prefetch_related('messages')


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ['timestamp']
    fields = ['message_type', 'content', 'timestamp', 'file_reference']


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session_user', 'message_type', 'content_preview', 'timestamp']
    list_filter = ['message_type', 'timestamp', 'session__user']
    search_fields = ['content', 'session__user__username', 'session__title']
    readonly_fields = ['timestamp']
    
    def session_user(self, obj):
        return obj.session.user.username
    session_user.short_description = "사용자"
    
    def content_preview(self, obj):
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content
    content_preview.short_description = "내용 미리보기"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('session__user', 'file_reference')


@admin.register(GeneratedReport)
class GeneratedReportAdmin(admin.ModelAdmin):
    list_display = [
        'title', 'user', 'report_type', 'status', 
        'file_size_mb', 'download_count', 'generation_date'
    ]
    list_filter = ['report_type', 'status', 'generation_date', 'user']
    search_fields = ['title', 'user__username', 'user_query']
    readonly_fields = [
        'generation_date', 'completion_date', 'file_size_mb', 
        'download_count', 'download_link'
    ]
    
    fieldsets = (
        ('기본 정보', {
            'fields': ('title', 'user', 'report_type', 'status')
        }),
        ('원본 파일', {
            'fields': ('source_file', 'user_query')
        }),
        ('생성된 파일', {
            'fields': (
                'output_filename', 'output_file_path', 'file_size', 'file_size_mb',
                'download_link', 'download_count'
            )
        }),
        ('내용', {
            'fields': ('markdown_content',),
            'classes': ('collapse',)
        }),
        ('날짜 정보', {
            'fields': ('generation_date', 'completion_date')
        }),
    )
    
    def download_link(self, obj):
        if obj.output_file_path and obj.status == 'completed':
            url = reverse('download_report') + f'?filename={obj.output_filename}'
            return format_html(
                '<a href="{}" target="_blank">파일 다운로드</a>', 
                url
            )
        return "파일 없음"
    download_link.short_description = "다운로드"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'source_file')


# ChatSession에 메시지 인라인 추가
ChatSessionAdmin.inlines = [ChatMessageInline]


# 관리자 사이트 커스터마이징
admin.site.site_header = "FlowMate AI 관리자"
admin.site.site_title = "FlowMate AI"
admin.site.index_title = "FlowMate AI 관리자 대시보드"
