import numpy as np


def simulate_panel_measurement(center, point, alpha_deg, beta_deg, gamma_deg, beta0_deg=30.0):
    """
    Вычисляет, какие азимут и угол места покажет полотно
    для одной точки в пространстве.

    Важно:
    - в глобальной системе штатное положение полотна уже имеет угол места beta0_deg
    - для самого полотна это штатное направление считается нулевым
    - alpha_deg, beta_deg, gamma_deg — это поправки относительно штатного положения
    """

    # 1. Переводим координаты в numpy-массивы
    center = np.array(center, dtype=float)
    point = np.array(point, dtype=float)

    # 2. Переводим углы из градусов в радианы
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)
    gamma = np.deg2rad(gamma_deg)
    beta0 = np.deg2rad(beta0_deg)

    # 3. Матрица штатного положения полотна
    # Полотно по умолчанию поднято на beta0 градусов по углу места
    R0 = np.array([
        [1, 0, 0],
        [0, np.cos(beta0), -np.sin(beta0)],
        [0, np.sin(beta0),  np.cos(beta0)]
    ], dtype=float)

    # 4. Матрица поправки
    R_err = np.array([
        [
            np.cos(alpha) * np.cos(gamma) + np.sin(alpha) * np.sin(beta) * np.sin(gamma),
            np.sin(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(gamma) - np.sin(alpha) * np.sin(beta) * np.cos(gamma)
        ],
        [
            -np.sin(alpha) * np.cos(gamma) + np.cos(alpha) * np.sin(beta) * np.sin(gamma),
            np.cos(alpha) * np.cos(beta),
            -np.sin(alpha) * np.sin(gamma) - np.cos(alpha) * np.sin(beta) * np.cos(gamma)
        ],
        [
            -np.cos(beta) * np.sin(gamma),
            np.sin(beta),
            np.cos(beta) * np.cos(gamma)
        ]
    ], dtype=float)

    # 5. Полная матрица ориентации полотна в глобальной системе
    R_total = R0 @ R_err

    # 6. Глобальный вектор от центра полотна к точке
    d = point - center

    # 7. Нормируем его
    u = d / np.linalg.norm(d)

    # 8. Переводим его в локальную систему полотна
    v = R_total.T @ u

    # 9. Считаем азимут и угол места
    a = np.arctan2(v[0], v[1])
    b = np.arcsin(v[2])

    # 10. Переводим в градусы
    a_deg = np.rad2deg(a)
    b_deg = np.rad2deg(b)

    return {
        "a_deg": a_deg,
        "b_deg": b_deg,
        "R0": R0,
        "R_err": R_err,
        "R_total": R_total,
        "u": u,
        "v": v
    }


if __name__ == "__main__":
    print("Программа моделирования измерения одной точки полотном")
    print("Штатное положение полотна: угол места 30 градусов в глобальной системе")
    print("Введите данные последовательно.\n")

    # Центр полотна
    xc = float(input("Центр полотна, X = "))
    yc = float(input("Центр полотна, Y = "))
    zc = float(input("Центр полотна, Z = "))

    # Точка
    x = float(input("\nТочка, X = "))
    y = float(input("Точка, Y = "))
    z = float(input("Точка, Z = "))

    # Поправки относительно штатного положения
    alpha_deg = float(input("\nПоправка alpha вокруг Z, градусы = "))
    beta_deg = float(input("Поправка beta вокруг X, градусы = "))
    gamma_deg = float(input("Поправка gamma вокруг Y, градусы = "))

    result = simulate_panel_measurement(
        center=[xc, yc, zc],
        point=[x, y, z],
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        gamma_deg=gamma_deg,
        beta0_deg=30.0
    )

    print("\nРезультаты:")
    print(f"Азимут a = {result['a_deg']:.6f} град")
    print(f"Угол места b = {result['b_deg']:.6f} град")

    print("\nГлобальный единичный вектор u:")
    print(result["u"])

    print("\nЛокальный единичный вектор v:")
    print(result["v"])

    print("\nМатрица штатного положения R0:")
    print(result["R0"])

    print("\nМатрица поправки R_err:")
    print(result["R_err"])

    print("\nПолная матрица ориентации R_total:")
    print(result["R_total"])