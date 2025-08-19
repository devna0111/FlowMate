"""
FlowMate 워크플로우 - 간소화된 상태 기반 처리
효율적인 의도 분류 → 전문 생성 → 품질 검증 파이프라인
"""

import re
import os
from typing import Dict, Any, Optional, List
from enum import Enum

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from vectordb_upload_search import data_to_vectorstore, BufferMemory, get_llm, ensure_korean_only
from utils.intent_classifier import check_intent, normalize_label
from utils.docx_writer import markdown_to_styled_docx
from utils.pptx_writer import save_structured_text_to_pptx


class TaskType(Enum):
    REPORT = "보고서"
    PRESENTATION = "발표자료"  
    SUMMARY = "요약"
    QA = "질의응답"
    HR = "HR평가"
    UNKNOWN = "기타"


class WorkflowState(BaseModel):
    """워크플로우 상태 관리"""
    # 입력
    query: str = ""
    file_path: str = ""
    memory: Optional[BufferMemory] = None
    
    # 중간 결과
    intent: str = ""
    task_type: TaskType = TaskType.UNKNOWN
    documents: List[str] = []
    raw_response: str = ""
    
    # 최종 결과
    final_response: str = ""
    output_file_path: Optional[str] = None
    success: bool = False
    error_message: str = ""
    
    class Config:
        arbitrary_types_allowed = True


class FlowMateWorkflow:
    def __init__(self):
        self.llm = get_llm(tokens=2048)
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """간소화된 순차 실행"""
        try:
            # 1. 의도 분류
            state = self.classify_intent(state)
            if state.error_message:
                return self.handle_error(state)
            
            # 2. 문서 검색
            state = self.retrieve_documents(state)
            if state.error_message:
                return self.handle_error(state)
            
            # 3. 태스크별 생성
            if state.task_type == TaskType.REPORT:
                state = self.generate_report(state)
            elif state.task_type == TaskType.PRESENTATION:
                state = self.generate_presentation(state)
            elif state.task_type == TaskType.SUMMARY:
                state = self.generate_summary(state)
            else:
                state = self.generate_qa_response(state)
            
            if state.error_message:
                return self.handle_error(state)
            
            # 4. 품질 검증
            state = self.check_quality(state)
            if state.error_message:
                return self.handle_error(state)
            
            # 5. 파일 생성 (필요한 경우)
            if state.task_type in [TaskType.REPORT, TaskType.PRESENTATION]:
                state = self.create_output_file(state)
            
            return state
            
        except Exception as e:
            state.error_message = f"워크플로우 실행 오류: {str(e)}"
            return self.handle_error(state)
    
    def classify_intent(self, state: WorkflowState) -> WorkflowState:
        """의도 분류"""
        try:
            # Intent classifier 사용
            check_intents = check_intent(state.query)
            result = self.llm.invoke(check_intents.format_messages(query=state.query))
            state.intent = normalize_label(result.content)
            
            # TaskType 매핑
            if "[보고서]" in state.intent:
                state.task_type = TaskType.REPORT
            elif "[발표]" in state.intent:
                state.task_type = TaskType.PRESENTATION
            elif "[요약]" in state.intent:
                state.task_type = TaskType.SUMMARY
            else:
                state.task_type = TaskType.QA
                
            print(f"[의도 분류] {state.intent} -> {state.task_type}")
            
        except Exception as e:
            state.error_message = f"의도 분류 실패: {str(e)}"
            state.task_type = TaskType.UNKNOWN
            
        return state
    
    def retrieve_documents(self, state: WorkflowState) -> WorkflowState:
        """문서 검색 및 벡터 스토어 활용"""
        try:
            vector_store = data_to_vectorstore(state.file_path)
            if vector_store:
                # 태스크별 검색 문서 수 조정
                k = 1000 if state.task_type in [TaskType.REPORT, TaskType.PRESENTATION] else 500
                docs = vector_store.similarity_search(state.query, k=k)
                state.documents = [doc.page_content for doc in docs]
                print(f"[문서 검색] {len(state.documents)}개 문서 검색됨")
            else:
                # 폴백: 직접 파일 읽기
                with open(state.file_path, 'r', encoding='utf-8') as f:
                    state.documents = [f.read()[:10000]]  # 10KB 제한
                print("[문서 검색] 폴백 모드로 파일 직접 읽기")
                
        except Exception as e:
            state.error_message = f"문서 검색 실패: {str(e)}"
            
        return state
    
    
    def generate_report(self, state: WorkflowState) -> WorkflowState:
        """보고서 생성"""
        try:
            system_prompt = self._get_korean_system_prompt()
            user_prompt = f"""
아래 문서 내용을 기반으로 전문적인 보고서를 마크다운 형식으로 작성해주세요.

문서 내용:
{chr(10).join(state.documents)}

사용자 요청: {state.query}

보고서 구조:
1. 제목
2. 목차  
3. 개요
4. 주요 내용 (섹션별)
5. 결론 및 권고사항
"""
            
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = self.llm.invoke(messages)
            state.raw_response = response.content
            print("[보고서 생성] 완료")
            
        except Exception as e:
            state.error_message = f"보고서 생성 실패: {str(e)}"
            
        return state
    
    def generate_presentation(self, state: WorkflowState) -> WorkflowState:
        """발표자료 생성"""
        try:
            system_prompt = self._get_korean_system_prompt()
            user_prompt = f"""
아래 문서 내용을 기반으로 PPT 슬라이드 구성을 작성해주세요.

문서 내용:
{chr(10).join(state.documents)}

사용자 요청: {state.query}

출력 형식:
[슬라이드 1]
제목: 제목 내용
핵심 포인트:
- 포인트 1
- 포인트 2

[슬라이드 2]
제목: 제목 내용
핵심 포인트:
- 포인트 1
- 포인트 2
"""
            
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = self.llm.invoke(messages)
            state.raw_response = response.content
            print("[발표자료 생성] 완료")
            
        except Exception as e:
            state.error_message = f"발표자료 생성 실패: {str(e)}"
            
        return state
    
    def generate_summary(self, state: WorkflowState) -> WorkflowState:
        """요약 생성"""
        try:
            system_prompt = self._get_korean_system_prompt()
            user_prompt = f"""
아래 문서 내용을 체계적으로 요약해주세요.

문서 내용:
{chr(10).join(state.documents)}

사용자 요청: {state.query}
"""
            
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = self.llm.invoke(messages)
            state.raw_response = response.content
            print("[요약 생성] 완료")
            
        except Exception as e:
            state.error_message = f"요약 생성 실패: {str(e)}"
            
        return state
    
    def generate_qa_response(self, state: WorkflowState) -> WorkflowState:
        """질의응답 생성"""
        try:
            system_prompt = self._get_korean_system_prompt()
            
            # 메모리 히스토리 포함
            history = ""
            if state.memory:
                history = state.memory.get_formatted_history()
            
            user_prompt = f"""
이전 대화:
{history}

문서 내용:
{chr(10).join(state.documents)}

사용자 질문: {state.query}

위 문서를 참고하여 사용자의 질문에 정확하고 친절하게 답변해주세요.
"""
            
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            response = self.llm.invoke(messages)
            state.raw_response = response.content
            print("[질의응답 생성] 완료")
            
        except Exception as e:
            state.error_message = f"질의응답 생성 실패: {str(e)}"
            
        return state
    
    def check_quality(self, state: WorkflowState) -> WorkflowState:
        """응답 품질 검증 및 한국어 번역"""
        try:
            if not state.raw_response:
                state.error_message = "생성된 응답이 없습니다."
                return state
                
            # ensure_korean_only 함수를 사용하여 품질 검증 및 번역
            verified_response = ensure_korean_only(state.raw_response)
            
            # 품질 검증 통과
            state.final_response = verified_response
            state.success = True
            print("[품질 검증 및 번역] 완료")
            
        except Exception as e:
            state.error_message = f"품질 검증 실패: {str(e)}"
            
        return state
    
    
    def create_output_file(self, state: WorkflowState) -> WorkflowState:
        """출력 파일 생성"""
        try:
            if state.task_type == TaskType.REPORT:
                file_path = "uploads/report_sample.docx"
                markdown_to_styled_docx(state.final_response, output_path=file_path)
                state.output_file_path = file_path
                print(f"[파일 생성] 보고서 파일 생성: {file_path}")
                
            elif state.task_type == TaskType.PRESENTATION:
                file_path = "uploads/pptx_sample.pptx"
                save_structured_text_to_pptx(state.final_response, output_path=file_path)
                state.output_file_path = file_path
                print(f"[파일 생성] 발표자료 파일 생성: {file_path}")
                
        except Exception as e:
            state.error_message = f"파일 생성 실패: {str(e)}"
            
        return state
    
    def handle_error(self, state: WorkflowState) -> WorkflowState:
        """에러 처리"""
        state.success = False
        state.final_response = f"죄송합니다. 처리 중 오류가 발생했습니다: {state.error_message}"
        print(f"[에러 처리] {state.error_message}")
        return state
    
    def _get_korean_system_prompt(self) -> str:
        """한국어 강제 시스템 프롬프트"""
        return """
[CRITICAL LANGUAGE INSTRUCTION]
반드시 한국어로만 답변하세요. 중국어, 영어, 일본어 등 다른 언어는 절대 사용 금지입니다.
ONLY Korean language allowed. Chinese/English/Japanese strictly forbidden.
只能用韩语回答，严禁使用中文或其他语言。

당신은 Flow팀에서 만든 FlowMate:사내업무길라잡이 AI입니다.
모든 답변은 반드시 한국어로만 작성해주세요.
전문적이고 정확한 내용을 한국어로 제공해주세요.
"""


# 전역 워크플로우 인스턴스
_workflow_instance = None

def get_workflow():
    """워크플로우 싱글톤 인스턴스"""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = FlowMateWorkflow()
    return _workflow_instance


def execute_workflow(query: str, file_path: str, memory: BufferMemory = None) -> WorkflowState:
    """워크플로우 실행"""
    workflow = get_workflow()
    
    # 초기 상태 설정
    initial_state = WorkflowState(
        query=query,
        file_path=file_path,
        memory=memory or BufferMemory()
    )
    
    # 워크플로우 실행
    result = workflow.execute(initial_state)
    
    # 메모리 업데이트 (QA의 경우)
    if result.success and result.task_type == TaskType.QA and memory:
        memory.append(query, result.final_response)
    
    return result