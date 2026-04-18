import numpy as np


ALPHA0_DEG = 0.0
BETA0_DEG = 30.0
GAMMA0_DEG = 0.0


def simulate_panel_measurement(alpha_deg, beta_deg, gamma_deg, x, y, z):
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    R_alpha = np.array(
        [
            [np.cos(alpha), np.sin(alpha), 0.0],
            [-np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    R_beta = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(beta), -np.sin(beta)],
            [0.0, np.sin(beta), np.cos(beta)],
        ]
    )

    R_gamma = np.array(
        [
            [np.cos(gamma), 0.0, np.sin(gamma)],
            [0.0, 1.0, 0.0],
            [-np.sin(gamma), 0.0, np.cos(gamma)],
        ]
    )

    R = R_alpha @ R_beta @ R_gamma

    d = np.array([x, y, z], dtype=float)
    u = d / np.linalg.norm(d)
    v = R.T @ u

    a = np.arctan2(v[0], v[1])
    b = np.arcsin(v[2])

    a_deg = np.degrees(a)
    b_deg = np.degrees(b)

    return a_deg, b_deg, R, u, v


def simulate_panel_measurement_from_corrections(d_alpha_deg, d_beta_deg, d_gamma_deg, x, y, z):
    alpha_deg = ALPHA0_DEG + d_alpha_deg
    beta_deg = BETA0_DEG + d_beta_deg
    gamma_deg = GAMMA0_DEG + d_gamma_deg
    return simulate_panel_measurement(alpha_deg, beta_deg, gamma_deg, x, y, z)


if __name__ == "__main__":
    print("Программа расчёта углов одной точки в локальной системе полотна")
    print("Введите поправки полотна и координаты точки в глобальной системе.")
    print()

    d_alpha_deg = float(input("Поправка dAlpha, градусы = "))
    d_beta_deg = float(input("Поправка dBeta, градусы = "))
    d_gamma_deg = float(input("Поправка dGamma, градусы = "))
    print()

    x = float(input("Точка, X = "))
    y = float(input("Точка, Y = "))
    z = float(input("Точка, Z = "))

    a_deg, b_deg, R, u, v = simulate_panel_measurement_from_corrections(
        d_alpha_deg,
        d_beta_deg,
        d_gamma_deg,
        x,
        y,
        z,
    )

    print()
    print("Результаты:")
    print(f"a_deg = {a_deg:.6f}")
    print(f"b_deg = {b_deg:.6f}")
    print()
    print("Матрица поворота R:")
    print(R)
    print("Глобальный единичный вектор u =", u)
    print("Локальный единичный вектор v =", v)
