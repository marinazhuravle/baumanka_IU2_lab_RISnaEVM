import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import find_GYR
import config

# 1. Загрузка и подготовка данных
data = pd.read_csv(config.file_path_mk, sep=' ')

data.drop([0, 1], inplace=True)
data.reset_index(drop=True, inplace=True)

gyro_data, error, columns_GYR = find_GYR.process_gyro_columns(df=data)
gyro_data['rate'] = data['rate']

# Убедимся, что time - числовой тип
if not pd.api.types.is_numeric_dtype(gyro_data['time']):
    gyro_data['time'] = pd.to_numeric(gyro_data['time'], errors='coerce')

# Убедимся, что rate - числовой тип
if not pd.api.types.is_numeric_dtype(gyro_data['rate']):
    gyro_data['rate'] = pd.to_numeric(gyro_data['rate'], errors='coerce')

for axis in [col for col in gyro_data.columns if col != ['time', 'rate']]:
    # Убедимся, что данные оси - числовые
    if not pd.api.types.is_numeric_dtype(gyro_data[axis]):
        gyro_data[axis] = pd.to_numeric(gyro_data[axis], errors='coerce')


split_data = {rate: group for rate, group in gyro_data.groupby('rate')}

if error:
    print(error)

# График "сырых" измерений (°/с)
plt.figure(figsize=(12, 6))
for axis in columns_GYR:
    plt.plot(gyro_data['time'], gyro_data[axis], label=axis)
plt.xlabel('Время, с')
plt.ylabel('Угловая скорость, °/с')
plt.title('Сырые измерения гироскопа от времени')
plt.grid(True)
plt.legend()
plt.savefig('Plots\Сырые_измерения_от_времени_мк.png')
plt.show()


# 1. Расчет масштабного коэффициента
def linear_trend(x, a, b):
    return a * x + b


def calculate_mk():
    m_koeff = {}
    zero_offset = {}
    print('\n')
    for ax in columns_GYR:
        # params - оптимальные значения параметров, при которых сумма квадратов невязок f(xdata, *popt) - ydata минимальна
        trend_rate = curve_fit(f=linear_trend, xdata=gyro_data['rate'], ydata=gyro_data[ax])[0]
        m_koeff[ax] = trend_rate[0] * 1000
        zero_offset[ax] = trend_rate[1] * m_koeff[ax] / 1000
        print(f"Масштабный коэффициент для {ax}: {m_koeff[ax]:.2f} мВ/(°/c)")
    return m_koeff, zero_offset


mk_koeff, zero_ofst = calculate_mk()


# 2. Расчет нелинейности
def calculate_nonlinearity(gyr_data):
    nonlinearity = {}
    deviations_data = {ax: [] for ax in columns_GYR}
    rates = sorted(split_data.keys())

    plt.figure(figsize=(12, 8))
    colors = ['r', 'g', 'b']

    for ax, color in zip(columns_GYR, colors):
        for rate in rates:
            group_df = split_data[rate]
            popt = curve_fit(lambda x, a, b: a * x + b, group_df['time'], group_df[ax])[0]
            linear_fit = popt[0] * group_df['time'] + popt[1]
            deviations = group_df[ax] - linear_fit
            deviations_data[ax].append(np.mean(deviations))

            max_deviation = np.max(np.abs(deviations))
            signal_range = np.max(gyr_data[ax]) - np.min(gyr_data[ax])
            nonlinearity[(ax, rate)] = (max_deviation / signal_range) * 100  # Кортеж как ключ

    for ax, color in zip(columns_GYR, colors):
        plt.plot(rates, deviations_data[ax], 'o-', color=color, label=f'{ax}', markersize=8, linewidth=2)

    plt.xlabel('Скорость, °/с', fontsize=12)
    plt.ylabel('Отклонение от линейности, В', fontsize=12)
    plt.title('Нелинейность гироскопа', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Plots/Графики_нелинейности_МК.png', dpi=300)
    plt.show()

    for axiss in columns_GYR:
        print(f'\nНелинейность для {axiss}:')
        for (ax, rate), value in nonlinearity.items():
            if ax==axiss:
                print(f'Скорость {rate} °/с: {value:.2f}%')

    return nonlinearity

non_lin = calculate_nonlinearity(gyro_data)


# 3. Расчет несимметричности
def calculate_asymmetry(col):
    """
    Расчет несимметричности масштабного коэффициента.
    Несимметричность = |МК_положительный - МК_отрицательный| / средний_МК * 100%
    В нашем случае все измерения при одной угловой скорости,
    поэтому используем отклонения от среднего.
    """
    print('\n')
    rates = sorted(split_data.keys())
    for ax in col:
        print(f'\nНесимметричность для {ax} °/с:')
        for rate in rates:
            group_df = split_data[rate]
            # Среднее значение выходного сигнала
            mean_value = np.mean(group_df[ax])

            # Положительные и отрицательные отклонения
            positive_dev = np.mean(group_df[ax][group_df[ax] > mean_value] - mean_value)
            negative_dev = np.mean(mean_value - group_df[ax][group_df[ax] < mean_value])

            # Рассчитываем несимметричность
            asymmetry = np.abs(positive_dev - negative_dev) / ((positive_dev + negative_dev) / 2) * 100
            print(f'Скорость {rate}: {asymmetry:.2f} %')
    return


calculate_asymmetry(col=columns_GYR)

# 4. Расчет смещения нуля
print('\n')
for axis in columns_GYR:
    print(f"Смещение нуля {axis}: {zero_ofst[axis]:.4f} °/с")
