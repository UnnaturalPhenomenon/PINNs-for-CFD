import os
import logging
import sys

def setup_logging(logger_name, level: int = logging.INFO) -> None:
    """ 
    프로그램에 사용할 통합 로거를 설정합니다.
    로그 포맷에 로거 이름을 포함시키고, 콘솔 출력 핸들러를 추가합니다.
    """
    #디렉토리 생성
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    #로그 파일 이름 설정
    log_filename = os.path.join(log_dir, f'{logger_name}.log')

    # 최상위 로거 가져오기.
    root_logger = logging.getLogger(logger_name)
    root_logger.setLevel(level)

    # 콘솔 핸들러 중복 방지
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )

    # 1. 파일 핸들러: 로그를 logs 폴더 내 __name__.log 파일에 저장합니다.
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 2. 콘솔 핸들러: 콘솔에 로그를 출력합니다.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger