import numpy as np


ALPHA0_DEG = 0.0
BETA0_DEG = 30.0
GAMMA0_DEG = 0.0


def normalize(vector):
    vector = np.array(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def vector_from_angles(a_deg, b_deg):
    a_rad = np.radians(a_deg)
    b_rad = np.radians(b_deg)
    return np.array(
        [
            np.sin(a_rad) * np.cos(b_rad),
            np.cos(a_rad) * np.cos(b_rad),
            np.sin(b_rad),
        ],
        dtype=float,
    )


def orthonormal_basis(first_vector, second_vector):
    e1 = first_vector
    t = second_vector - np.dot(second_vector, first_vector) * first_vector
    e2 = t / np.linalg.norm(t)
    e3 = np.cross(e1, e2)
    return np.column_stack((e1, e2, e3))


def estimate_panel_corrections(P1, P2, a1_deg, b1_deg, a2_deg, b2_deg):
    d1 = np.array(P1, dtype=float)
    d2 = np.array(P2, dtype=float)

    u1 = normalize(d1)
    u2 = normalize(d2)

    v1 = vector_from_angles(a1_deg, b1_deg)
    v2 = vector_from_angles(a2_deg, b2_deg)

    E_l = orthonormal_basis(v1, v2)
    E_g = orthonormal_basis(u1, u2)

    R = E_g @ E_l.T

    beta_rad = np.arcsin(R[2, 1])
    alpha_rad = np.arctan2(R[0, 1], R[1, 1])
    gamma_rad = np.arctan2(-R[2, 0], R[2, 2])

    alpha_deg = np.degrees(alpha_rad)
    beta_deg = np.degrees(beta_rad)
    gamma_deg = np.degrees(gamma_rad)

    dAlpha_deg = alpha_deg - ALPHA0_DEG
    dBeta_deg = beta_deg - BETA0_DEG
    dGamma_deg = gamma_deg - GAMMA0_DEG

    return {
        "alpha_deg": alpha_deg,
        "beta_deg": beta_deg,
        "gamma_deg": gamma_deg,
        "dAlpha_deg": dAlpha_deg,
        "dBeta_deg": dBeta_deg,
        "dGamma_deg": dGamma_deg,
        "R": R,
    }


def print_result(result):
    print()
    print("Результаты расчёта:")
    print(f"alpha_deg  = {result['alpha_deg']:.6f}")
    print(f"beta_deg   = {result['beta_deg']:.6f}")
    print(f"gamma_deg  = {result['gamma_deg']:.6f}")
    print(f"dAlpha_deg = {result['dAlpha_deg']:.6f}")
    print(f"dBeta_deg  = {result['dBeta_deg']:.6f}")
    print(f"dGamma_deg = {result['dGamma_deg']:.6f}")
    print()
    print("Матрица поворота R:")
    print(result["R"])


def main():
    print("Программа расчёта поправок антенного полотна по двум точкам")
    print("Введите данные последовательно.")
    print()

    x1 = float(input("Точка 1, X = "))
    y1 = float(input("Точка 1, Y = "))
    z1 = float(input("Точка 1, Z = "))
    a1_deg = float(input("Точка 1, азимут a, градусы = "))
    b1_deg = float(input("Точка 1, угол места b, градусы = "))
    print()

    x2 = float(input("Точка 2, X = "))
    y2 = float(input("Точка 2, Y = "))
    z2 = float(input("Точка 2, Z = "))
    a2_deg = float(input("Точка 2, азимут a, градусы = "))
    b2_deg = float(input("Точка 2, угол места b, градусы = "))

    P1 = np.array([x1, y1, z1], dtype=float)
    P2 = np.array([x2, y2, z2], dtype=float)

    result = estimate_panel_corrections(P1, P2, a1_deg, b1_deg, a2_deg, b2_deg)
    print_result(result)


if __name__ == "__main__":
    main()
