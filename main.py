import os

plots_dir = 'Plots'

# Проверяем существование папки и создаем при необходимости
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

print('\n\nПРОАНАЛИЗИРУЕМ ДРЕЙФ ГИРОСКОПА\n')
from RISnaEVM import drift

print('\n\nПРОАНАЛИЗИРУЕМ МАСШТАБНЫЙ КОЭФФИЦИЕНТ\n')
from RISnaEVM import scale_factor