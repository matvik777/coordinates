import numpy as np
import math

def estimate_panel_misalignment(a1_deg, b1_deg, p1, a2_deg, b2_deg, p2):
    """
    Estimate the installation error angles of an antenna panel from two corner reflectors.

    Parameters:
    a1_deg (float): Azimuth angle to reflector 1 in degrees
    b1_deg (float): Elevation angle to reflector 1 in degrees
    p1 (list or np.array): Global coordinates of reflector 1 [x1, y1, z1]
    a2_deg (float): Azimuth angle to reflector 2 in degrees
    b2_deg (float): Elevation angle to reflector 2 in degrees
    p2 (list or np.array): Global coordinates of reflector 2 [x2, y2, z2]

    Returns:
    tuple: (alpha_deg, beta_deg, gamma_deg, R) where R is the rotation matrix
    """
    # Convert inputs to numpy arrays
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    # Convert angles to radians
    a1 = math.radians(a1_deg)
    b1 = math.radians(b1_deg)
    a2 = math.radians(a2_deg)
    b2 = math.radians(b2_deg)

    # Step 1: Build local unit vectors from measured angles
    v1 = np.array([
        math.sin(a1) * math.cos(b1),
        math.cos(a1) * math.cos(b1),
        math.sin(b1)
    ])
    v2 = np.array([
        math.sin(a2) * math.cos(b2),
        math.cos(a2) * math.cos(b2),
        math.sin(b2)
    ])

    # Step 2: Build global unit vectors from reflector coordinates
    d1 = p1
    d2 = p2

    # Check for zero norm
    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)
    if norm_d1 == 0 or norm_d2 == 0:
        raise ValueError("Reflector coordinates must not be at the origin.")

    u1 = d1 / norm_d1
    u2 = d2 / norm_d2

    # Step 3: Recover the rotation matrix
    # Local basis
    e1_l = v1
    proj = np.dot(v2, v1) * v1
    t_l = v2 - proj
    norm_t_l = np.linalg.norm(t_l)
    if norm_t_l < 1e-6:
        raise ValueError("Local vectors are nearly collinear.")
    e2_l = t_l / norm_t_l
    e3_l = np.cross(e1_l, e2_l)
    E_l = np.column_stack([e1_l, e2_l, e3_l])

    # Global basis
    e1_g = u1
    proj_g = np.dot(u2, u1) * u1
    t_g = u2 - proj_g
    norm_t_g = np.linalg.norm(t_g)
    if norm_t_g < 1e-6:
        raise ValueError("Global vectors are nearly collinear.")
    e2_g = t_g / norm_t_g
    e3_g = np.cross(e1_g, e2_g)
    E_g = np.column_stack([e1_g, e2_g, e3_g])

    # Rotation matrix
    R = E_g @ E_l.T

    # Optional orthogonality check
    RTR = R.T @ R
    det_R = np.linalg.det(R)
    if not np.allclose(RTR, np.eye(3), atol=1e-6) or not np.isclose(det_R, 1.0, atol=1e-6):
        raise ValueError("Computed rotation matrix is not orthogonal or has wrong determinant.")

    # Step 4: Extract angles
    beta = math.asin(R[2, 1])  # R[2,1] in 0-based indexing
    alpha = math.atan2(R[0, 1], R[1, 1])
    gamma = math.atan2(-R[2, 0], R[2, 2])

    # Convert to degrees
    alpha_deg = math.degrees(alpha)
    beta_deg = math.degrees(beta)
    gamma_deg = math.degrees(gamma)

    return alpha_deg, beta_deg, gamma_deg, R

# Demonstration example
if __name__ == "__main__":
    print("Программа для оценки ошибки установки антенной панели на основе двух уголковых отражателей.")
    print("=" * 70)

    try:
        # Ввод данных для первого отражателя
        print("\nВведите данные для первого уголкового отражателя:")
        a1_deg = float(input("Азимут (в градусах): "))
        b1_deg = float(input("Угол места (в градусах): "))
        x1 = float(input("Координата X: "))
        y1 = float(input("Координата Y: "))
        z1 = float(input("Координата Z: "))
        p1 = [x1, y1, z1]

        # Ввод данных для второго отражателя
        print("\nВведите данные для второго уголкового отражателя:")
        a2_deg = float(input("Азимут (в градусах): "))
        b2_deg = float(input("Угол места (в градусах): "))
        x2 = float(input("Координата X: "))
        y2 = float(input("Координата Y: "))
        z2 = float(input("Координата Z: "))
        p2 = [x2, y2, z2]

        # Вызов функции
        alpha, beta, gamma, R = estimate_panel_misalignment(a1_deg, b1_deg, p1, a2_deg, b2_deg, p2)

        # Вывод результатов
        print("\n" + "=" * 70)
        print("Результаты оценки ошибки установки антенной панели:")
        print(f"Азимутальная ошибка (alpha): {alpha:.2f} градусов")
        print(f"Ошибка по углу места (beta): {beta:.2f} градусов")
        print(f"Ошибка поворота вокруг нормали (gamma): {gamma:.2f} градусов")
        print("\nМатрица поворота R:")
        print(R)

    except ValueError as e:
        print(f"\nОшибка ввода или расчета: {e}")
        print("Проверьте введенные данные (должны быть числа) и убедитесь, что векторы не коллинеарны и координаты не в начале координат.")
    except Exception as e:
        print(f"\nНеожиданная ошибка: {e}")

    # Дополнительные тесты (опционально, закомментированы)
    # print("\n--- Тестовые случаи ---")
    # ... (старые тесты)