from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def generate_feedback(summary: str, audio_features: dict, visual_features: dict) -> str:
    prompt_template = ChatPromptTemplate.from_template("""
[발표 요약]
{summary}

[음성 분석 결과]
- 총 길이: {duration_sec}초
- 평균 피치: {avg_pitch}
- 억양 변화량: {energy_variation}
- 말의 속도 (추정): {speech_tempo}

[시각 표현 분석 결과]
- 얼굴 인식 비율: {face_detection_ratio}
- 제스처 사용 비율: {gesture_ratio}

위 발표 데이터를 바탕으로 다음 기준에 따라 **아주 구체적이고 실전적인 발표 코칭 피드백**을 작성해줘.

1. 발표의 장점 3가지와, 개선해야 할 점 3가지 이상을 반드시 구분해서 제시해줘.  
2. 각 항목마다 실제 상황에서 바로 쓸 수 있는 개선 방법, 예시, 연습 팁을 자세히 설명해줘.  
3. 음성·시각적 수치는 실제 발표 현장에서 어떻게 해석되는지 근거와 함께 설명해줘.  
4. 발표 목적(정보전달/설득/보고 등)을 추론하여 해당 목적에 적합한 전달 전략을 추가 조언해줘.
5. 청중(일반/전문가/학생 등) 유형을 추정해, 청중에 맞춘 구체적인 어투/내용 개선 팁을 추가해줘.
6. 마지막에 한 문장으로 동기부여되는 발표 팁을 따로 강조해줘.
7. 분석 결과값이 0에 가까우면 ‘영상 품질 문제’ 안내 문구를 반드시 포함해줘.

아래 예시와 같이 상세하게 작성해줘.
[예시]
- 장점1: ~ / 구체적 칭찬 + 근거
- 개선점1: ~ / 구체적 개선 방법 + 연습 팁
...
- [한 문장 조언] ~

(꼭 2,000자 이상으로 작성해줘)
""")
    llm = ChatOllama(model="qwen2.5vl:7b")  # 적절한 모델로 교체 가능
    chain = prompt_template | llm
    return chain.invoke({
        **audio_features,
        **visual_features,
        "summary": summary
    }).content
    
