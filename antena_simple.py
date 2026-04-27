import numpy as np


def estimate_panel_orientation(center, point1, point2, a1_deg, b1_deg, a2_deg, b2_deg):
    """
    Оценивает ориентацию антенного полотна по двум точкам.
    """

    # 1. Координаты -> numpy-массивы
    center = np.array(center, dtype=float)
    point1 = np.array(point1, dtype=float)
    point2 = np.array(point2, dtype=float)

    # 2. Углы: градусы -> радианы
    a1 = np.deg2rad(a1_deg)
    b1 = np.deg2rad(b1_deg)
    a2 = np.deg2rad(a2_deg)
    b2 = np.deg2rad(b2_deg)

    # 3. Локальные единичные векторы
    v1 = np.array([
        np.sin(a1) * np.cos(b1),
        np.cos(a1) * np.cos(b1),
        np.sin(b1)
    ], dtype=float)

    v2 = np.array([
        np.sin(a2) * np.cos(b2),
        np.cos(a2) * np.cos(b2),
        np.sin(b2)
    ], dtype=float)

    # 4. Глобальные векторы от центра к точкам
    d1 = point1 - center
    d2 = point2 - center

    # 5. Глобальные единичные векторы
    u1 = d1 / np.linalg.norm(d1)
    u2 = d2 / np.linalg.norm(d2)

    # 6. Локальный базис
    e1_l = v1
    t_l = v2 - np.dot(v2, v1) * v1
    e2_l = t_l / np.linalg.norm(t_l)
    e3_l = np.cross(e1_l, e2_l)
    E_l = np.column_stack((e1_l, e2_l, e3_l))

    # 7. Глобальный базис
    e1_g = u1
    t_g = u2 - np.dot(u2, u1) * u1
    e2_g = t_g / np.linalg.norm(t_g)
    e3_g = np.cross(e1_g, e2_g)
    E_g = np.column_stack((e1_g, e2_g, e3_g))

    # 8. Матрица поворота
    R = E_g @ E_l.T
    beta0 = np.deg2rad(30.0)

    R0 = np.array([
        [1, 0, 0],
        [0, np.cos(beta0), -np.sin(beta0)],
        [0, np.sin(beta0),  np.cos(beta0)]
    ], dtype=float)

    R_err = R0.T @ R

    beta_err = np.arcsin(R_err[2, 1])
    alpha_err = np.arctan2(R_err[0, 1], R_err[1, 1])
    gamma_err = np.arctan2(-R_err[2, 0], R_err[2, 2])

    dAlpha_deg = np.rad2deg(alpha_err)
    dBeta_deg = np.rad2deg(beta_err)
    dGamma_deg = np.rad2deg(gamma_err)

    return {
        "R_err": R_err,
        "dAlpha_deg": dAlpha_deg,
        "dBeta_deg": dBeta_deg,
        "dGamma_deg": dGamma_deg,
        "v1": v1,
        "v2": v2,
        "u1": u1,
        "u2": u2,
    }


if __name__ == "__main__":
    print("Программа определения ориентации полотна по двум точкам")
    print("Задайте данные в коде ниже.\n")

    # Центр полотна
    xc = 0.0
    yc = 0.0
    zc = 33.0

    # Точка 1
    x1 = -280.0
    y1 = 500.0
    z1 = -3.0
    a1 = -30.9
    b1 = -34.45

    # Точка 2
    x2 = 350
    y2 = 760
    z2 = 0
    a2 = 33.45
    b2 = -29.43

    result = estimate_panel_orientation(
        center=[xc, yc, zc],
        point1=[x1, y1, z1],
        point2=[x2, y2, z2],
        a1_deg=a1,
        b1_deg=b1,
        a2_deg=a2,
        b2_deg=b2
    )

    print("\nПоправки относительно штатного положения:")
    print(f"dAlpha_deg = {result['dAlpha_deg']:.6f}")
    print(f"dBeta_deg  = {result['dBeta_deg']:.6f}")
    print(f"dGamma_deg = {result['dGamma_deg']:.6f}")

    print("\nМатрица поворота ошибки R_err:")
    print(result["R_err"])
