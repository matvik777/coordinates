import numpy as np
import matplotlib.pyplot as plt


def vector_from_angles(a_deg, b_deg):
    """
    Строит локальный единичный вектор по азимуту и углу места.
    Формула полностью совпадает с той, что используется в текущих файлах проекта.
    """

    a = np.deg2rad(a_deg)
    b = np.deg2rad(b_deg)

    return np.array([
        np.sin(a) * np.cos(b),
        np.cos(a) * np.cos(b),
        np.sin(b)
    ], dtype=float)


def normalize_vector(vector, error_message):
    """
    Нормирует вектор и выбрасывает понятную ошибку,
    если норма слишком мала.
    """

    norm_value = np.linalg.norm(vector)
    if norm_value <= 1e-12:
        raise ValueError(error_message)

    return vector / norm_value


def build_correction_matrix(d_alpha_deg, d_beta_deg, d_gamma_deg):
    """
    Строит матрицу поправки так же, как это делается в perevod.py:
    R_err = R_alpha @ R_beta @ R_gamma
    """

    da = np.deg2rad(d_alpha_deg)
    db = np.deg2rad(d_beta_deg)
    dg = np.deg2rad(d_gamma_deg)

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

    return R_alpha @ R_beta @ R_gamma


def simulate_panel_measurement(center, point, alpha_deg, beta_deg, gamma_deg, beta0_deg=30.0):
    """
    Генерирует "истинные" углы для одной точки так же,
    как это делает ogle.py.

    На вход подаются:
    - точный центр полотна
    - точная глобальная координата точки
    - истинные поправки полотна

    На выходе получаем идеальные азимут и угол места,
    без какого-либо шума по углам.
    """

    center = np.array(center, dtype=float)
    point = np.array(point, dtype=float)

    beta0 = np.deg2rad(beta0_deg)

    # Штатное положение полотна: в глобальной системе уже есть угол места 30 градусов.
    R0 = np.array([
        [1, 0, 0],
        [0, np.cos(beta0), -np.sin(beta0)],
        [0, np.sin(beta0),  np.cos(beta0)]
    ], dtype=float)

    # Матрица поправки полотна.
    R_err = build_correction_matrix(alpha_deg, beta_deg, gamma_deg)

    # Полная ориентация полотна в глобальной системе.
    R_total = R0 @ R_err

    # Глобальный вектор от центра полотна к точке.
    d = point - center
    u = normalize_vector(d, "Точка совпадает с центром полотна.")

    # Переход в локальную систему полотна.
    v = R_total.T @ u

    a_deg = np.rad2deg(np.arctan2(v[0], v[1]))
    b_deg = np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0)))

    return {
        "a_deg": a_deg,
        "b_deg": b_deg,
        "R0": R0,
        "R_err": R_err,
        "R_total": R_total,
        "u": u,
        "v": v
    }


def estimate_panel_orientation(center, point1, point2, a1_deg, b1_deg, a2_deg, b2_deg, beta0_deg=30.0):
    """
    Восстанавливает поправки полотна по двум точкам.
    Математика совпадает с antena_simple.py.
    """

    center = np.array(center, dtype=float)
    point1 = np.array(point1, dtype=float)
    point2 = np.array(point2, dtype=float)

    # Локальные единичные векторы, построенные по идеальным углам.
    v1 = vector_from_angles(a1_deg, b1_deg)
    v2 = vector_from_angles(a2_deg, b2_deg)

    # Глобальные единичные векторы, построенные по координатам точек.
    d1 = point1 - center
    d2 = point2 - center
    u1 = normalize_vector(d1, "Первая точка совпадает с центром полотна.")
    u2 = normalize_vector(d2, "Вторая точка совпадает с центром полотна.")

    # Локальный базис по двум лучам.
    e1_l = v1
    t_l = v2 - np.dot(v2, v1) * v1
    e2_l = normalize_vector(t_l, "Локальные лучи почти коллинеарны.")
    e3_l = np.cross(e1_l, e2_l)
    E_l = np.column_stack((e1_l, e2_l, e3_l))

    # Глобальный базис по двум лучам.
    e1_g = u1
    t_g = u2 - np.dot(u2, u1) * u1
    e2_g = normalize_vector(t_g, "Глобальные лучи почти коллинеарны.")
    e3_g = np.cross(e1_g, e2_g)
    E_g = np.column_stack((e1_g, e2_g, e3_g))

    # Восстановленная полная матрица поворота.
    R = E_g @ E_l.T

    beta0 = np.deg2rad(beta0_deg)
    R0 = np.array([
        [1, 0, 0],
        [0, np.cos(beta0), -np.sin(beta0)],
        [0, np.sin(beta0),  np.cos(beta0)]
    ], dtype=float)

    # Выделяем только матрицу ошибки относительно штатного положения.
    R_err = R0.T @ R

    # Из матрицы ошибки извлекаем углы поправки.
    beta_err = np.arcsin(np.clip(R_err[2, 1], -1.0, 1.0))
    alpha_err = np.arctan2(R_err[0, 1], R_err[1, 1])
    gamma_err = np.arctan2(-R_err[2, 0], R_err[2, 2])

    return {
        "R_err": R_err,
        "dAlpha_deg": np.rad2deg(alpha_err),
        "dBeta_deg": np.rad2deg(beta_err),
        "dGamma_deg": np.rad2deg(gamma_err),
        "v1": v1,
        "v2": v2,
        "u1": u1,
        "u2": u2,
    }


def add_coordinate_noise(point, noise_sigma, rng):
    """
    Добавляет гауссов шум только к декартовым координатам точки.

    Здесь noise_sigma трактуется как СКО ошибки
    по каждой координате X, Y, Z в метрах.
    """

    point = np.array(point, dtype=float)

    if noise_sigma <= 0:
        return point.copy()

    return point + rng.normal(loc=0.0, scale=noise_sigma, size=3)


def wrap_angle_error_deg(estimated_deg, true_deg):
    """
    Разность углов с приведением к диапазону [-180, 180].
    Это защищает от ложных скачков на 360 градусов.
    """

    return (estimated_deg - true_deg + 180.0) % 360.0 - 180.0


def is_valid_point_pair(center, point1, point2, min_angle_between_points_deg):
    """
    Проверяет, что точки дают устойчивую геометрию:
    - обе точки не совпадают с центром
    - направления на точки не почти совпадают
    """

    center = np.array(center, dtype=float)
    point1 = np.array(point1, dtype=float)
    point2 = np.array(point2, dtype=float)

    d1 = point1 - center
    d2 = point2 - center

    if np.linalg.norm(d1) <= 1e-12 or np.linalg.norm(d2) <= 1e-12:
        return False

    u1 = d1 / np.linalg.norm(d1)
    u2 = d2 / np.linalg.norm(d2)

    cosine_value = np.clip(np.dot(u1, u2), -1.0, 1.0)
    angle_between_deg = np.rad2deg(np.arccos(cosine_value))

    return angle_between_deg >= min_angle_between_points_deg


def generate_random_point(bounds, rng):
    """
    Генерирует одну истинную точку в заданном прямоугольном диапазоне координат.
    """

    return np.array([
        rng.uniform(bounds["x_min"], bounds["x_max"]),
        rng.uniform(bounds["y_min"], bounds["y_max"]),
        rng.uniform(bounds["z_min"], bounds["z_max"]),
    ], dtype=float)


def build_true_trials(
    center,
    n_trials,
    rng,
    use_fixed_points,
    fixed_point1,
    fixed_point2,
    point_bounds,
    min_angle_between_points_deg,
    d_alpha_true_deg,
    d_beta_true_deg,
    d_gamma_true_deg,
    beta0_deg,
):
    """
    Подготавливает "истинные" данные для всех Monte Carlo прогонов.

    Для каждого прогона сохраняем:
    - две истинные точки
    - идеальные углы, сгенерированные через simulate_panel_measurement

    Эти данные потом используются повторно для всех уровней шума,
    чтобы уровни шума сравнивались на одной и той же геометрии.
    """

    trials = []

    if use_fixed_points:
        point1_true = np.array(fixed_point1, dtype=float)
        point2_true = np.array(fixed_point2, dtype=float)

        if not is_valid_point_pair(center, point1_true, point2_true, min_angle_between_points_deg):
            raise ValueError("Фиксированные точки дают слишком плохую геометрию для восстановления.")

        for _ in range(n_trials):
            measurement1 = simulate_panel_measurement(
                center=center,
                point=point1_true,
                alpha_deg=d_alpha_true_deg,
                beta_deg=d_beta_true_deg,
                gamma_deg=d_gamma_true_deg,
                beta0_deg=beta0_deg,
            )
            measurement2 = simulate_panel_measurement(
                center=center,
                point=point2_true,
                alpha_deg=d_alpha_true_deg,
                beta_deg=d_beta_true_deg,
                gamma_deg=d_gamma_true_deg,
                beta0_deg=beta0_deg,
            )

            trials.append({
                "point1_true": point1_true.copy(),
                "point2_true": point2_true.copy(),
                "a1_true_deg": measurement1["a_deg"],
                "b1_true_deg": measurement1["b_deg"],
                "a2_true_deg": measurement2["a_deg"],
                "b2_true_deg": measurement2["b_deg"],
            })

        return trials

    for _ in range(n_trials):
        for _attempt in range(1000):
            point1_true = generate_random_point(point_bounds, rng)
            point2_true = generate_random_point(point_bounds, rng)

            if not is_valid_point_pair(center, point1_true, point2_true, min_angle_between_points_deg):
                continue

            measurement1 = simulate_panel_measurement(
                center=center,
                point=point1_true,
                alpha_deg=d_alpha_true_deg,
                beta_deg=d_beta_true_deg,
                gamma_deg=d_gamma_true_deg,
                beta0_deg=beta0_deg,
            )
            measurement2 = simulate_panel_measurement(
                center=center,
                point=point2_true,
                alpha_deg=d_alpha_true_deg,
                beta_deg=d_beta_true_deg,
                gamma_deg=d_gamma_true_deg,
                beta0_deg=beta0_deg,
            )

            trials.append({
                "point1_true": point1_true,
                "point2_true": point2_true,
                "a1_true_deg": measurement1["a_deg"],
                "b1_true_deg": measurement1["b_deg"],
                "a2_true_deg": measurement2["a_deg"],
                "b2_true_deg": measurement2["b_deg"],
            })
            break
        else:
            raise RuntimeError("Не удалось сгенерировать устойчивую пару точек в заданных пределах.")

    return trials


def compute_error_statistics(errors):
    """
    Считает статистику по массиву угловых ошибок.
    """

    errors = np.asarray(errors, dtype=float)

    if errors.size == 0:
        return {
            "mean_abs": np.nan,
            "rmse": np.nan,
            "std": np.nan,
            "mean": np.nan,
        }

    return {
        "mean_abs": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "std": float(np.std(errors, ddof=1)) if errors.size > 1 else 0.0,
        "mean": float(np.mean(errors)),
    }


def run_monte_carlo_experiment(
    center,
    d_alpha_true_deg,
    d_beta_true_deg,
    d_gamma_true_deg,
    noise_levels,
    n_trials,
    rng,
    use_fixed_points,
    fixed_point1,
    fixed_point2,
    point_bounds,
    min_angle_between_points_deg=5.0,
    beta0_deg=30.0,
    max_noise_attempts=20,
):
    """
    Основной Monte Carlo эксперимент.

    Для каждого уровня шума:
    - берём две истинные точки и идеальные углы
    - добавляем шум только в координаты
    - по шумным координатам и идеальным углам восстанавливаем поправки
    - сравниваем найденные углы с истинными
    """

    trials = build_true_trials(
        center=center,
        n_trials=n_trials,
        rng=rng,
        use_fixed_points=use_fixed_points,
        fixed_point1=fixed_point1,
        fixed_point2=fixed_point2,
        point_bounds=point_bounds,
        min_angle_between_points_deg=min_angle_between_points_deg,
        d_alpha_true_deg=d_alpha_true_deg,
        d_beta_true_deg=d_beta_true_deg,
        d_gamma_true_deg=d_gamma_true_deg,
        beta0_deg=beta0_deg,
    )

    results = []

    for noise_sigma in noise_levels:
        alpha_errors = []
        beta_errors = []
        gamma_errors = []
        skipped_trials = 0

        for trial in trials:
            success = False

            for _attempt in range(max_noise_attempts):
                # Шум добавляется только к координатам двух точек.
                point1_noisy = add_coordinate_noise(trial["point1_true"], noise_sigma, rng)
                point2_noisy = add_coordinate_noise(trial["point2_true"], noise_sigma, rng)

                try:
                    estimated = estimate_panel_orientation(
                        center=center,
                        point1=point1_noisy,
                        point2=point2_noisy,
                        a1_deg=trial["a1_true_deg"],
                        b1_deg=trial["b1_true_deg"],
                        a2_deg=trial["a2_true_deg"],
                        b2_deg=trial["b2_true_deg"],
                        beta0_deg=beta0_deg,
                    )
                except ValueError:
                    continue

                alpha_errors.append(wrap_angle_error_deg(estimated["dAlpha_deg"], d_alpha_true_deg))
                beta_errors.append(wrap_angle_error_deg(estimated["dBeta_deg"], d_beta_true_deg))
                gamma_errors.append(wrap_angle_error_deg(estimated["dGamma_deg"], d_gamma_true_deg))
                success = True
                break

            if not success:
                skipped_trials += 1

        alpha_errors = np.asarray(alpha_errors, dtype=float)
        beta_errors = np.asarray(beta_errors, dtype=float)
        gamma_errors = np.asarray(gamma_errors, dtype=float)

        alpha_stats = compute_error_statistics(alpha_errors)
        beta_stats = compute_error_statistics(beta_errors)
        gamma_stats = compute_error_statistics(gamma_errors)

        results.append({
            "noise_sigma": float(noise_sigma),
            "valid_trials": int(alpha_errors.size),
            "skipped_trials": int(skipped_trials),
            "alpha_errors": alpha_errors,
            "beta_errors": beta_errors,
            "gamma_errors": gamma_errors,
            "alpha_stats": alpha_stats,
            "beta_stats": beta_stats,
            "gamma_stats": gamma_stats,
        })

    return results, trials


def print_results_table(results):
    """
    Печатает таблицу результатов по каждому уровню шума.
    """

    header = (
        " noise_m | valid | skipped | mae_alpha | mae_beta | mae_gamma | "
        "rmse_alpha | rmse_beta | rmse_gamma | std_alpha | std_beta | std_gamma "
    )
    print(header)
    print("-" * len(header))

    for result in results:
        alpha_stats = result["alpha_stats"]
        beta_stats = result["beta_stats"]
        gamma_stats = result["gamma_stats"]

        print(
            f"{result['noise_sigma']:8.2f} | "
            f"{result['valid_trials']:5d} | "
            f"{result['skipped_trials']:7d} | "
            f"{alpha_stats['mean_abs']:9.4f} | "
            f"{beta_stats['mean_abs']:8.4f} | "
            f"{gamma_stats['mean_abs']:9.4f} | "
            f"{alpha_stats['rmse']:10.4f} | "
            f"{beta_stats['rmse']:9.4f} | "
            f"{gamma_stats['rmse']:10.4f} | "
            f"{alpha_stats['std']:9.4f} | "
            f"{beta_stats['std']:8.4f} | "
            f"{gamma_stats['std']:9.4f}"
        )


def plot_mean_absolute_errors(results):
    """
    Строит 3 subplot:
    влияние уровня шума на среднюю абсолютную ошибку
    по dAlpha, dBeta, dGamma.
    """

    noise_levels = [result["noise_sigma"] for result in results]
    alpha_mae = [result["alpha_stats"]["mean_abs"] for result in results]
    beta_mae = [result["beta_stats"]["mean_abs"] for result in results]
    gamma_mae = [result["gamma_stats"]["mean_abs"] for result in results]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle("Влияние ошибки координат на среднюю абсолютную ошибку поправок")

    axes[0].plot(noise_levels, alpha_mae, marker="o", linewidth=2)
    axes[0].set_ylabel("MAE dAlpha, град")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(noise_levels, beta_mae, marker="o", linewidth=2)
    axes[1].set_ylabel("MAE dBeta, град")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(noise_levels, gamma_mae, marker="o", linewidth=2)
    axes[2].set_ylabel("MAE dGamma, град")
    axes[2].set_xlabel("СКО шума координат, м")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_rmse_curves(results):
    """
    Строит общий график RMSE для трёх поправок.
    """

    noise_levels = [result["noise_sigma"] for result in results]
    alpha_rmse = [result["alpha_stats"]["rmse"] for result in results]
    beta_rmse = [result["beta_stats"]["rmse"] for result in results]
    gamma_rmse = [result["gamma_stats"]["rmse"] for result in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(noise_levels, alpha_rmse, marker="o", linewidth=2, label="RMSE dAlpha")
    ax.plot(noise_levels, beta_rmse, marker="s", linewidth=2, label="RMSE dBeta")
    ax.plot(noise_levels, gamma_rmse, marker="^", linewidth=2, label="RMSE dGamma")
    ax.set_title("RMSE восстановления поправок")
    ax.set_xlabel("СКО шума координат, м")
    ax.set_ylabel("RMSE, град")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_histograms(results, histogram_noise_level):
    """
    Для одного уровня шума строит гистограммы ошибок
    по dAlpha, dBeta, dGamma.
    """

    available_levels = np.array([result["noise_sigma"] for result in results], dtype=float)
    selected_index = int(np.argmin(np.abs(available_levels - histogram_noise_level)))
    selected_result = results[selected_index]
    selected_level = selected_result["noise_sigma"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Распределение ошибок при шуме координат {selected_level:.2f} м")

    axes[0].hist(selected_result["alpha_errors"], bins=25, edgecolor="black", alpha=0.8)
    axes[0].set_title("Ошибка dAlpha")
    axes[0].set_xlabel("град")
    axes[0].set_ylabel("Частота")
    axes[0].grid(True, alpha=0.2)

    axes[1].hist(selected_result["beta_errors"], bins=25, edgecolor="black", alpha=0.8)
    axes[1].set_title("Ошибка dBeta")
    axes[1].set_xlabel("град")
    axes[1].grid(True, alpha=0.2)

    axes[2].hist(selected_result["gamma_errors"], bins=25, edgecolor="black", alpha=0.8)
    axes[2].set_title("Ошибка dGamma")
    axes[2].set_xlabel("град")
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # ==================================================
    # ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
    # ==================================================

    # Центр полотна считаем известным точно.
    center = np.array([0.0, 0.0, 33], dtype=float)

    # Истинные поправки полотна.
    dAlpha_true_deg = -2.0
    dBeta_true_deg = 3.0
    dGamma_true_deg = 4.0

    # Штатный угол места полотна в глобальной системе.
    beta0_deg = 30.0

    # Режим задания истинных точек.
    # Если True, используются две фиксированные точки ниже.
    # Если False, для каждого Monte Carlo прогона генерируется своя истинная пара точек.
    use_fixed_points = False

    fixed_point1 = np.array([-250.0, 620.0, 40.0], dtype=float)
    fixed_point2 = np.array([180.0, 650.0, 120.0], dtype=float)

    # Диапазоны для генерации реалистичных точек.
    # По Y точки находятся примерно на дальности около 600 метров,
    # по X в секторе примерно от -350 до +350 метров,
    # по Z диапазон задаётся отдельно.
    point_bounds = {
        "x_min": -350.0,
        "x_max": 350.0,
        "y_min": 550.0,
        "y_max": 700.0,
        "z_min": -10.0,
        "z_max": 10.0,
    }

    # Минимальный угол между направлениями на две точки.
    # Это помогает избежать почти вырожденной геометрии.
    min_angle_between_points_deg = 10.0

    # Уровни шума координат.
    # Каждый уровень интерпретируется как СКО гауссова шума
    # по каждой из координат X, Y, Z.
    noise_levels = [0, 1, 2, 5, 10, 15, 20]

    # Число Monte Carlo прогонов на каждый уровень шума.
    n_trials = 200

    # Для какого уровня шума строить гистограммы ошибок.
    histogram_noise_level = 10.0

    # Фиксируем seed для воспроизводимости эксперимента.
    random_seed = 42
    rng = np.random.default_rng(random_seed)

    print("Monte Carlo эксперимент по влиянию ошибки GPS-координат на восстановление поправок полотна")
    print(f"Истинные поправки: dAlpha={dAlpha_true_deg:.3f} deg, dBeta={dBeta_true_deg:.3f} deg, dGamma={dGamma_true_deg:.3f} deg")
    print(f"Штатный угол места полотна: beta0={beta0_deg:.3f} deg")
    print(f"Число прогонов на уровень шума: {n_trials}")
    print(f"Seed генератора случайных чисел: {random_seed}")

    if use_fixed_points:
        print("Режим точек: фиксированные точки")
        print(f"point1 = {fixed_point1}")
        print(f"point2 = {fixed_point2}")
    else:
        print("Режим точек: случайные точки в заданном секторе")
        print(f"bounds = {point_bounds}")

    results, trials = run_monte_carlo_experiment(
        center=center,
        d_alpha_true_deg=dAlpha_true_deg,
        d_beta_true_deg=dBeta_true_deg,
        d_gamma_true_deg=dGamma_true_deg,
        noise_levels=noise_levels,
        n_trials=n_trials,
        rng=rng,
        use_fixed_points=use_fixed_points,
        fixed_point1=fixed_point1,
        fixed_point2=fixed_point2,
        point_bounds=point_bounds,
        min_angle_between_points_deg=min_angle_between_points_deg,
        beta0_deg=beta0_deg,
    )

    print("\nТаблица результатов:")
    print_results_table(results)

    # Строим графики влияния шума координат на ошибку восстановления поправок.
    plot_mean_absolute_errors(results)
    plot_rmse_curves(results)
    plot_error_histograms(results, histogram_noise_level=histogram_noise_level)

    plt.show()
