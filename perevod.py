import numpy as np


def correct_measurement_to_nominal(a_bad_deg, b_bad_deg, d_alpha_deg, d_beta_deg, d_gamma_deg):
    """
    Переводит измерение из локальной системы кривого полотна
    в локальную систему ровного полотна.
    """

    # 1. Переводим все углы в радианы
    a_bad = np.deg2rad(a_bad_deg)
    b_bad = np.deg2rad(b_bad_deg)

    da = np.deg2rad(d_alpha_deg)
    db = np.deg2rad(d_beta_deg)
    dg = np.deg2rad(d_gamma_deg)

    # 2. Строим вектор в системе кривого полотна
    v_bad = np.array([
        np.sin(a_bad) * np.cos(b_bad),
        np.cos(a_bad) * np.cos(b_bad),
        np.sin(b_bad)
    ], dtype=float)

    # 3. Строим матрицы поправки
    R_alpha = np.array([
        [np.cos(da),  np.sin(da), 0],
        [-np.sin(da), np.cos(da), 0],
        [0,           0,          1]
    ], dtype=float)

    R_beta = np.array([
        [1, 0, 0],
        [0, np.cos(db), -np.sin(db)],
        [0, np.sin(db),  np.cos(db)]
    ], dtype=float)

    R_gamma = np.array([
        [np.cos(dg), 0, np.sin(dg)],
        [0,          1, 0],
        [-np.sin(dg), 0, np.cos(dg)]
    ], dtype=float)

    # 4. Полная матрица поправки
    R_err = R_alpha @ R_beta @ R_gamma

    # 5. Переводим вектор из кривого полотна в ровное
    v_nom = R_err @ v_bad

    # 6. Из исправленного вектора считаем новые углы
    a_nom = np.arctan2(v_nom[0], v_nom[1])
    b_nom = np.arcsin(v_nom[2])

    # 7. Переводим в градусы
    a_nom_deg = np.rad2deg(a_nom)
    b_nom_deg = np.rad2deg(b_nom)

    return {
        "a_nom_deg": a_nom_deg,
        "b_nom_deg": b_nom_deg,
        "v_bad": v_bad,
        "v_nom": v_nom,
        "R_err": R_err
    }


if __name__ == "__main__":
    print("Программа перевода измерения из кривого полотна в ровное")
    print("Задайте данные в коде ниже.\n")

    # Что дало кривое полотно
    a_bad_deg = 10.0
    b_bad_deg = 5.0

    # Найденные поправки
    d_alpha_deg = 1.0
    d_beta_deg = 2.0
    d_gamma_deg = 3.0

    result = correct_measurement_to_nominal(
        a_bad_deg=a_bad_deg,
        b_bad_deg=b_bad_deg,
        d_alpha_deg=d_alpha_deg,
        d_beta_deg=d_beta_deg,
        d_gamma_deg=d_gamma_deg
    )

    print("\nРезультаты:")
    print(f"Исправленный азимут a_nom = {result['a_nom_deg']:.6f} град")
    print(f"Исправленный угол места b_nom = {result['b_nom_deg']:.6f} град")

    print("\nВектор в системе кривого полотна v_bad:")
    print(result["v_bad"])

    print("\nВектор в системе ровного полотна v_nom:")
    print(result["v_nom"])

    print("\nМатрица поправки R_err:")
    print(result["R_err"])
