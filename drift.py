import numpy as np
import pandas as pd
import math
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import find_GYR
import config

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# 1. Загрузка и подготовка данных
data = pd.read_csv(config.file_path_dreif, sep=' ')

data.drop([0, 1], inplace=True)
data.reset_index(drop=True, inplace=True)


gyro_data, error, columns_GYR = find_GYR.process_gyro_columns(df=data)

# Убедимся, что time - числовой тип
if not pd.api.types.is_numeric_dtype(gyro_data['time']):
    gyro_data['time'] = pd.to_numeric(gyro_data['time'], errors='coerce')

for axis in [col for col in gyro_data.columns if col != 'time']:
    # Убедимся, что данные оси - числовые
    if not pd.api.types.is_numeric_dtype(gyro_data[axis]):
        gyro_data[axis] = pd.to_numeric(gyro_data[axis], errors='coerce')

# 2. График "сырых" измерений (°/с) - проверено
plt.figure(figsize=(12, 6))
for axis in columns_GYR:
    plt.plot(gyro_data['time'], gyro_data[axis], label=axis)
plt.xlabel('Время, с')
plt.ylabel('Угловая скорость, °/с')
plt.title('Сырые измерения гироскопа')
plt.grid(True)
plt.legend()
plt.savefig('Plots\Сырые_измерения_дрейф.png')
plt.show()

# 3. Расчет смещения нуля (bias)
# 4. Расчет тренда (линейный дрейф)
def linear_trend(x, a, b):
    return a * x + b

for axis in columns_GYR:
    # params - оптимальные значения параметров, при которых сумма квадратов невязок f(xdata, *popt) - ydata минимальна
    trend_rate = curve_fit(f=linear_trend, xdata=gyro_data['time'], ydata=gyro_data[axis])[0]
    # Тренд в °/с^2
    # Переводим в °/ч/ч: умножаем на 3600*3600
    trend_rate_hrhr = trend_rate[0] * 3600 * 3600
    bias = trend_rate[1]
    print(f"\nТренд для {axis}: {trend_rate_hrhr:.6f} °/ч/ч")
    print(f"Смещение нуля {axis}: {bias:.6f} °/с")


# 5. Расчет девиации Аллана
def allan_deviation(omega, fs, tau):
    """
    Расчет девиации Аллана
    :param omega: массив угловых скоростей (°/с)
    :param fs: частота дискретизации (Гц)
    :param tau: массив временных интервалов для расчета (с)
    :return: массив значений девиации Аллана для каждого tau
    """
    n = len(omega)
    adev = np.zeros_like(tau, dtype=float)

    for i, m in enumerate(tau):
        m = int(m * fs)  # Количество точек в интервале m
        if m == 0:
            m = 1
        d = n // m  # Количество групп

        # Разбиваем данные на группы по m точек
        groups = omega[:d * m].reshape(d, m)

        # Вычисляем средние для каждой группы
        group_means = groups.mean(axis=1)

        # Разности между соседними средними
        diffs = np.diff(group_means)

        # Девиация Аллана
        adev[i] = np.sqrt(0.5 * np.mean(diffs ** 2).astype(float))

    return adev


# Частота дискретизации (из данных - 100 Гц)
fs = 1 / (gyro_data['time'][1] - gyro_data['time'][0])

# Временные интервалы для расчета (логарифмическая шкала)
tau = np.logspace(-2, np.log10(len(gyro_data) / fs / 2), 100)

# Расчет для каждой оси
adev_results = {}
for axis in columns_GYR:
    adev = allan_deviation(gyro_data[axis].values, fs, tau)
    adev_results[axis] = adev

# 6. График девиации Аллана
plt.figure(figsize=(12, 6))
for axis in columns_GYR:
    plt.loglog(tau, adev_results[axis], label=axis)
plt.xlabel('Временной интервал, τ (с)')
plt.ylabel('Девиация Аллана, σ(τ) (°/с)')
plt.title('Девиация Аллана')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig('Plots\Девиация_Аллана_дрейф.png')
plt.show()


# 7. Аппроксимация девиации Аллана
def allan_variance_model(tau, N, B, K):
    """
    Модель для аппроксимации девиации Аллана
    :param tau: временной интервал
    :param N: коэффициент Angular Random Walk
    :param B: коэффициент Bias Instability
    :param K: коэффициент Rate Random Walk
    :return: значение девиации Аллана
    """
    return np.sqrt(N ** 2 / tau + B ** 2 * 2 / math.pi * math.log(2) + K ** 2 * tau / 3)


# Аппроксимация для каждой оси
params_results = {}
plt.figure(figsize=(12, 6))

for axis in columns_GYR:
    # Начальные приближения для параметров - стартовые точки для итерационного подбора
    p0 = [1e-3, 1e-2, 1e-3]
    print('\n')
    # Аппроксимация
    try:
        # подгон кривой на набор данных
        params = curve_fit(allan_variance_model, tau, adev_results[axis], p0=p0)[0]
        params_results[axis] = params

        # Расчет аппроксимированной кривой
        adev_fit = allan_variance_model(tau, *params)

        # График
        plt.loglog(tau, adev_results[axis], 'o', markersize=4, label=f'{axis} (данные)')
        plt.loglog(tau, adev_fit, label=f'{axis} (аппроксимация)')

        # Вывод параметров
        print(f"Параметры для {axis}:")
        print(f"Angular Random Walk (N): {params[0]:.6f} °/√c")
        print(f"Bias Instability (B): {params[1]:.6f} °/c")
        print(f"Rate Random Walk (K): {params[2]:.6f} °/c^(3 / 2)")

    except Exception as e:
        print(f"Ошибка при аппроксимации {axis}: {str(e)}")

plt.xlabel('Временной интервал, τ (с)')
plt.ylabel('Девиация Аллана, σ(τ) (°/с)')
plt.title('Девиация Аллана с аппроксимацией')
plt.grid(True, which="both", ls="-")
plt.savefig('Plots\Девиация_Аллана_с_аппроксимацией_дрейф.png')
plt.show()