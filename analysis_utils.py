import numpy as np

def safe_corrcoef(array1, array2):
    """
    Calculate correlation coefficient in a safe way:
      1) Remove NaN/Inf entries
      2) Skip if less than 2 data points remain
      3) Skip if either array has zero variance
    """
    arr1 = np.array(array1, dtype=float)
    arr2 = np.array(array2, dtype=float)

    # 1) NaN 또는 Inf 제거 (둘 중 하나라도 NaN/Inf면 제외)
    valid_mask = np.isfinite(arr1) & np.isfinite(arr2)
    arr1_clean = arr1[valid_mask]
    arr2_clean = arr2[valid_mask]

    # 2) 길이가 2 미만이면 상관계수 계산 불가
    if len(arr1_clean) < 2 or len(arr2_clean) < 2:
        print("Not enough data to calculate correlation.")
        return None

    # 3) 분산=0 (모두 동일값)이면 상관계수 계산 불가
    if np.std(arr1_clean) == 0 or np.std(arr2_clean) == 0:
        print("분산=0인 데이터가 있어 상관계수 계산 불가.")
        return None

    # 4) 안전한 corrcoef
    cc = np.corrcoef(arr1_clean, arr2_clean)[0, 1]
    return cc 