from without_domain import *

def data_preparation_without_domain():
    titanic = load_titanic_dataset()
    # [데이터 준비 방법]
    # - Variable Selection : 결측치가 많고 불필요한 컬럼은 제거해주세요
    variable_selection(titanic)
    
    # - Handling Missing Values : 결측치가 있으면 채워주세요
    handling_missing_values(titanic)
    
    # - Vectorization : 컴퓨터가 이해할 수 있도록 데이터를 변환해주세요
    vectorization_sex(titanic)
    vectorization_embarked(titanic)

    
    # [실습] 
    # 아래에 실습 코드를 입력하여
    # 데이터가 어떻게 처리되는지 확인하세요.
    
    # [Bonus] 결측치(Null)를 확인해주세요.
    check_missing_values(titanic)

    # 데이터 처리 결과를 보여줍니다
    show_result(titanic)
    
if __name__ == "__main__":
    data_preparation_without_domain()