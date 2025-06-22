# Находим колонки с GYR в названии
def process_gyro_columns(df):

    gyro_cols = [col for col in df.columns if 'GYR' in col.upper()]

    error_message = None

    # Проверяем количество колонок
    if len(gyro_cols) > 3:
        error_message = f"Ошибка: найдено {len(gyro_cols)} колонок с GYR, что превышает максимально допустимые 3. Будут использованы только первые 3 колонки."
        gyro_cols = gyro_cols[:3]

    # Создаем новый DataFrame с временем и GYR колонками
    result_data = df[['time'] + gyro_cols].copy()

    # Переименовываем колонки в зависимости от их количества
    new_names = ['GYR_X', 'GYR_Y', 'GYR_Z'][:len(gyro_cols)]
    rename_dict = {old: new for old, new in zip(gyro_cols, new_names)}
    result_data = result_data.rename(columns=rename_dict)

    gyro_columns = [col for col in result_data.columns if col != 'time']

    return result_data, error_message, gyro_columns