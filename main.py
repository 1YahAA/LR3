import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import trapz, simps
import math
# Определение функций
def f1(x):
    return x**3 + 3*x**2 - 9*x - 5

def f2(x):
    return -np.cos(x)

#найти точки пересечения
def find_peresech(f1, f2, x_start, x_end):
    peresech = []
    array_x = np.linspace(x_start, x_end, 500)
    for i in range(1, len(array_x)):
        root = fsolve(lambda x: f1(x) - f2(x), array_x[i])

        print(f"Root at x={array_x[i]}: {root}")

        for sol in root:
            if abs(f1(sol) - f2(sol)) < 1e-5:
                peresech.append((sol, f1(sol)))
    return peresech

#получение точек пересечения
peresech = find_peresech(f1, f2, -5, 5)
peresech_x = [point[0] for point in peresech]
peresech_y = [point[1] for point in peresech]

#вычислить значений функций
array_x = np.linspace(-5, 5, 500)
y1 = f1(array_x)
y2 = f2(array_x)

#построить графики
plt.figure(figsize=(10, 5))
plt.plot(array_x, y1, label='y = x^3 + 3x^2 - 9x - 5')
plt.plot(array_x, y2, label='y = -cos(x)')
plt.scatter(peresech_x, peresech_y, color='black', label='Точки пересечения')

#заштриховать замкнутые области
plt.fill_between(array_x, y1, y2, where=(y1 >= y2), color='blue', alpha=0.5, interpolate=True)# label='Заштрихованная область y+')
plt.fill_between(array_x, y1, y2, where=(y1 <= y2), color='blue', alpha=0.5, interpolate=True)# label='Заштрихованная область y-')

#вывод координет точек пересечения
for i in range(len(peresech_x)):
    plt.text(peresech_x[i], peresech_y[i], f'({peresech_x[i]:.2f}, {peresech_y[i]:.2f})', ha='right', va='bottom')



#находим площадь двумя способами
index = np.where(np.diff(np.sign(y1 - y2)) != 0)[0]
zone1_trapz = trapz(abs(y1[:index[0]] - y2[:index[0]]), x=array_x[:index[0]])
zone2_trapz = trapz(abs(y1[index[0]:] - y2[index[0]:]), x=array_x[index[0]:])

zone1_simpson = simps(abs(y1[:index[0]] - y2[:index[0]]), x=array_x[:index[0]])
zone2_simpson = simps(abs(y1[index[0]:] - y2[index[0]:]), x=array_x[index[0]:])

#вывод значений
plt.text(-3, 100, f"Площадь 1 (метод трапеций): {zone1_trapz:.2f}", fontsize=10, color='blue')
plt.text(-3, 90, f"Площадь 2 (метод трапеций): {zone2_trapz:.2f}", fontsize=10, color='blue')

plt.text(-3, 75, f"Площадь 1 (метод Симпсона): {zone1_simpson:.2f}", fontsize=10, color='red')
plt.text(-3, 65, f"Площадь 2 (метод Симпсона): {zone2_simpson:.2f}", fontsize=10, color='red')



plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()

