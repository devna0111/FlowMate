# def hr_predict(age, project, salary, number_of_turnovers, surround_eval, personal_history, edu_trips_lastyear, currentyear_at_company, buisnesstrip,when_enroll) :
#     '''
#     [나이, 참여프로젝트 수, 월급, 이직횟수, 주변평가(1~4),경력,
#     전년도교육출장횟수, 현회사근속년수, 출장횟수, 입사나이]
#     를 입력받아 pretrained_RandomForestClassifier 결과값을 반환하는 함수
#     '''
#     import joblib
#     import os
#     from django.conf import settings
    
#     # 필요 model, scaler import
#     base_dir = getattr(settings, 'BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#     model = joblib.load(os.path.join(base_dir, "rf_model_precision1.joblib"))
#     scaler = joblib.load(os.path.join(base_dir, "rf_model_precision1_scaler.joblib"))
    
#     나이=age; 참여프로젝트=project; 월급=salary; 이직회수=number_of_turnovers; 주변평가=surround_eval;
#     경력=personal_history; 전년도교육출장횟수=edu_trips_lastyear; 현회사근속년수=currentyear_at_company
#     출장횟수=buisnesstrip ; 입사나이 = when_enroll
#     출장_등급 = 0 if 출장횟수==0 else 1 if 1<=출장횟수<=29 else 2
#     근속연차 = 현회사근속년수 - 입사나이
#     이직률 = 이직회수 / 경력
#     프로젝트참여율 = 참여프로젝트 / 경력
#     교육출장참여율 = 전년도교육출장횟수 / 경력
#     현직근속비율 = 현회사근속년수 / 경력
#     연봉_경력비율 = 월급*12/경력
#     연봉_프로젝트비율 = 월급*12/참여프로젝트
#     경력_근속연차 = 경력-근속연차
#     근속연차차이 = 근속연차 - 현회사근속년수
#     프로젝트밀도지수 = 참여프로젝트+전년도교육출장횟수 / 경력
#     평판_근속년수 = 주변평가 * 현회사근속년수
#     연봉_평판점수 = 월급*12*주변평가
#     경력_나이비율 = 경력/나이
#     근속_나이비율 = 현회사근속년수/나이
#     연봉_나이 = 월급*12/나이
#     입사나이 = 나이 - 경력
#     출장odm = 출장_등급
#     x_col = [[나이, 참여프로젝트, 월급, 이직회수, 주변평가, 경력, 전년도교육출장횟수, 현회사근속년수,출장_등급, 이직률, 프로젝트참여율, 교육출장참여율, 현직근속비율, 연봉_경력비율,연봉_프로젝트비율, 경력_근속연차, 근속연차차이, 프로젝트밀도지수, 평판_근속년수, 연봉_평판점수,경력_나이비율, 근속_나이비율, 연봉_나이, 입사나이, 출장odm]]
#     result = model.predict(scaler.transform(x_col))
#     return "우수" if result != 0 else "보통"
# 
# if __name__ == "__main__" :
#     sample = hr_predict(age=34,
#                         project=3,
#                         salary=5838750,
#                         number_of_turnovers=1,
#                         surround_eval=3,
#                         personal_history=5,
#                         edu_trips_lastyear=2,
#                         currentyear_at_company=5,
#                         buisnesstrip=2,
#                         when_enroll=29)
#     print(sample)
import joblib
import pandas as pd
import os
import numpy as np

package = joblib.load('final_model.pkl')
model   = package["model"]
th      = package["threshold"]

# ===== 안전 나눗셈 =====
def safe_div(a, b):
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
        return a / b
    else:
        return np.nan if b == 0 else a / b

# ===== 파생변수 생성 함수 =====
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1차 파생
    out["월급_경력비"]    = safe_div(out["월급_KRW"], (out["경력"] + 1))
    out["월급_프로젝트비"] = safe_div(out["월급_KRW"], (out["참여프로젝트"] + 1))
    out["현근속_총경력비"] = safe_div(out["현회사근속년수"], (out["경력"] + 1))
    out["총근속"]     = out["현회사근속년수"] + (out["근속연차"] - out["현회사근속년수"])
    out["경력성장률"] = safe_div(out["현회사근속년수"], (out["근속연차"] + 1))
    out["이직빈도"]  = safe_div(out["이직회수"], (out["경력"] + 1))
    out["학습몰입도"] = out["참여프로젝트"] * out["전년도교육출장횟수"]

    # 2차 파생
    out["월급_근속연차비"]   = safe_div(out["월급_KRW"], (out["근속연차"] + 1))
    out["월급_출장비"]       = safe_div(out["월급_KRW"], (out["출장"] + 1))
    out["근속_경력비"]       = safe_div(out["현회사근속년수"], (out["경력"] + 1))
    out["평균근속기간"]       = safe_div(out["총근속"], (out["이직회수"] + 1))
    out["프로젝트_경력비"]     = safe_div(out["참여프로젝트"], (out["경력"] + 1))
    out["프로젝트_근속연차비"] = safe_div(out["참여프로젝트"], (out["근속연차"] + 1))
    out["출장_경력비"]        = safe_div(out["출장"], (out["경력"] + 1))
    out["출장_근속연차비"]     = safe_div(out["출장"], (out["근속연차"] + 1))
    out["교육출장_경력비"]     = safe_div(out["전년도교육출장횟수"], (out["경력"] + 1))
    out["주변평가_근속연차"]   = out["주변평가"] * out["근속연차"]
    out["학습몰입도_교육출장"] = out["학습몰입도"] * out["전년도교육출장횟수"]

    return out

# ===== 예측 함수 =====
def hr_predict(record: dict) -> str:
    '''
    "출장": int -> 출장횟수,
    "전년도교육출장횟수": int -> 1,
    "이직회수": int -> 0,
    "참여프로젝트": int -> 3,
    "월급_KRW": int -> 5000000,
    "경력": int -> 5,
    "현회사근속년수": int -> 5,
    "근속연차": int -> 5,
    "주변평가": int -> [1 ~ 4],
    "부서": {"R&D":0, "영업":1, "사무직":2},
    "전공": "공학계열", "자연과학계열", "사회과학계열","인문학","기타"
    "직급관리자여부": 1 if 관리직 이상 else 0
    '''
    df = pd.DataFrame([record])

    # 파생변수 자동 생성
    df_full = add_features(df)

    # 모델 예측
    prob = model.predict_proba(df_full)[:, 1][0]
    pred = int(prob >= th)

    return "우수" if pred == 1 else "보통"

# ===== 로컬 테스트 =====
if __name__ == "__main__":
    sample = {
        "출장": 2,
        "전년도교육출장횟수": 1,
        "이직회수": 0,
        "참여프로젝트": 3,
        "월급_KRW": 5000000,
        "경력": 5,
        "현회사근속년수": 5,
        "근속연차": 5,
        "주변평가": 3,
        "부서": 1,
        "전공": "공학계열",
        "직급관리자여부": 0
    }
    print(hr_predict(sample))
