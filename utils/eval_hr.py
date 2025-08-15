def hr_predict(age, project, salary, number_of_turnovers, surround_eval, personal_history, edu_trips_lastyear, currentyear_at_company, buisnesstrip,when_enroll) :
    '''
    [나이, 참여프로젝트 수, 월급, 이직횟수, 주변평가(1~4),경력,
    전년도교육출장횟수, 현회사근속년수, 출장횟수, 입사나이]
    를 입력받아 pretrained_RandomForestClassifier 결과값을 반환하는 함수
    '''
    import joblib
    import os
    from django.conf import settings
    
    # 필요 model, scaler import
    base_dir = getattr(settings, 'BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model = joblib.load(os.path.join(base_dir, "rf_model_precision1.joblib"))
    scaler = joblib.load(os.path.join(base_dir, "rf_model_precision1_scaler.joblib"))
    
    나이=age; 참여프로젝트=project; 월급=salary; 이직회수=number_of_turnovers; 주변평가=surround_eval;
    경력=personal_history; 전년도교육출장횟수=edu_trips_lastyear; 현회사근속년수=currentyear_at_company
    출장횟수=buisnesstrip ; 입사나이 = when_enroll
    출장_등급 = 0 if 출장횟수==0 else 1 if 1<=출장횟수<=29 else 2
    근속연차 = 현회사근속년수 - 입사나이
    이직률 = 이직회수 / 경력
    프로젝트참여율 = 참여프로젝트 / 경력
    교육출장참여율 = 전년도교육출장횟수 / 경력
    현직근속비율 = 현회사근속년수 / 경력
    연봉_경력비율 = 월급*12/경력
    연봉_프로젝트비율 = 월급*12/참여프로젝트
    경력_근속연차 = 경력-근속연차
    근속연차차이 = 근속연차 - 현회사근속년수
    프로젝트밀도지수 = 참여프로젝트+전년도교육출장횟수 / 경력
    평판_근속년수 = 주변평가 * 현회사근속년수
    연봉_평판점수 = 월급*12*주변평가
    경력_나이비율 = 경력/나이
    근속_나이비율 = 현회사근속년수/나이
    연봉_나이 = 월급*12/나이
    입사나이 = 나이 - 경력
    출장odm = 출장_등급
    x_col = [[나이, 참여프로젝트, 월급, 이직회수, 주변평가, 경력, 전년도교육출장횟수, 현회사근속년수,출장_등급, 이직률, 프로젝트참여율, 교육출장참여율, 현직근속비율, 연봉_경력비율,연봉_프로젝트비율, 경력_근속연차, 근속연차차이, 프로젝트밀도지수, 평판_근속년수, 연봉_평판점수,경력_나이비율, 근속_나이비율, 연봉_나이, 입사나이, 출장odm]]
    result = model.predict(scaler.transform(x_col))
    return "우수" if result != 0 else "보통"

if __name__ == "__main__" :
    sample = hr_predict(age=34,
                        project=3,
                        salary=5838750,
                        number_of_turnovers=1,
                        surround_eval=3,
                        personal_history=5,
                        edu_trips_lastyear=2,
                        currentyear_at_company=5,
                        buisnesstrip=2,
                        when_enroll=29)
    print(sample)