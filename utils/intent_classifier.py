# ✅ LangChain용 파인튜닝 지향 프롬프트 (few-shot 포함)
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b-instruct")
# 1) 예시 프롬프트 (입력/라벨 페어)
def check_intent(query) :
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{query}"),
        ("ai", "{label}")
    ])

    # 2) 학습과 추론 모두에서 사용할 대표 예시들 (동의어/우회 표현 다양화)
    examples = [
        # 보고서
        {"query": "이번 분기 판매 데이터 기반으로 매출 보고서 틀 좀 만들어줘", "label": "[보고서]"},
        {"query": "사내 결재용 레포트 양식으로 정리해줘",                    "label": "[보고서]"},
        {"query": "분석 결과를 문서형 보고 형태로 작성해",                    "label": "[보고서]"},
        # 요약
        {"query": "이 문서 핵심만 간단히 정리해줘",                           "label": "[요약]"},
        {"query": "기사 내용을 요점만 추려서 알려줘",                          "label": "[요약]"},
        {"query": "긴 텍스트를 한 단락으로 압축해줘",                          "label": "[요약]"},
        # 발표
        {"query": "발표자료(PPT) 형식으로 만들어줘",                           "label": "[발표]"},
        {"query": "슬라이드 10장 분량의 프레젠테이션 구성해줘",                "label": "[발표]"},
        {"query": "데크(deck) 초안으로 만들어줄래?",                           "label": "[발표]"},
        # 일반
        {"query": "내일 비와?",                                              "label": "[일반]"},
        {"query": "파이썬에서 리스트를 정렬하는 방법 알려줘",                  "label": "[일반]"},
        {"query": "Qdrant를 온라인으로 바꾸는 방법",                           "label": "[일반]"},
    ]

    fewshot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    # 3) 최종 프롬프트
    check_intent = ChatPromptTemplate.from_messages([
        ("system",
        "다음 규칙을 반드시 지켜라.\n"
        "1) 출력은 오직 아래 중 하나의 라벨 단 한 개만 반환: [보고서], [요약], [발표], [일반]\n"
        "2) 부가 설명, 따옴표, 공백, 마침표, 접두/접미 텍스트 금지.\n"
        "3) 매핑 기준:\n"
        "   - 보고서/레포트/문서 양식 요청 ⇒ [보고서]\n"
        "   - 요약/핵심정리/압축 요청 ⇒ [요약]\n"
        "   - 발표/슬라이드/PPT/프레젠테이션/데크 요청 ⇒ [발표]\n"
        "   - 위에 해당하지 않으면 ⇒ [일반]\n"),
        fewshot,
        ("human", "{query}")
    ])
    return check_intent

# 4) 간단한 후처리(모델이 규칙을 어겼을 때 대비)
VALID = {"[보고서]", "[요약]", "[발표]", "[일반]"}
def normalize_label(model_output: str) -> str:
    if not model_output:
        return "[일반]"
    text = model_output.strip()
    # 흔한 일탈 정리
    # 예: "라벨: [요약]" -> "[요약]"; "요약" -> "[요약]"
    if text in VALID:
        return text
    # 대괄호 없이 온 경우 매핑
    bare = text.replace("라벨:", "").replace("label:", "").strip()
    bare = bare.replace(" ", "")
    mapping = {"보고서":"[보고서]","요약":"[요약]","발표":"[발표]","일반":"[일반]"}
    if bare in mapping:
        return mapping[bare]
    # 라벨 패턴 추출
    import re
    m = re.search(r"\[(보고서|요약|발표|일반)\]", text)
    if m:
        return f"[{m.group(1)}]"
    return "[일반]"

# 사용 예시 (LLM 실행부는 환경에 맞춰 연결)
if __name__ == "__main__" :
    while True :
        query = input('명령어를 입력하세요 : ')
        if query == "끝" :
            break
        check_intents = check_intent(query)
        result = llm.invoke(check_intents.format_messages(query=query))
        label = normalize_label(result.content)
        print(label)  # -> [발표]

