# fmt:off
import numpy as np
from scipy.stats import f

data = np.array([
    [6.014, 7.907, 6.628, 7.882],
    [7.305, 7.514, 8.079, 8.219],
    [8.764, 8.51, 9.255, 9.773]
])
print(data)

# Обчислюємо загальне середнє значення
grand_mean = np.mean(data)

# Обчислюємо суму квадратів для фактору A (рядки)
ss_a = np.sum((np.mean(data, axis=1) - grand_mean) ** 2) * data.shape[1]

# Обчислюємо суму квадратів для фактору B (стовпці)
ss_b = np.sum((np.mean(data, axis=0) - grand_mean) ** 2) * data.shape[0]

# Обчислюємо суму квадратів для взаємодії факторів A і B
ss_ab = np.sum((data - np.mean(data, axis=0) - np.mean(data, axis=1)[:, np.newaxis] + grand_mean) ** 2)

# Обчислюємо ступені свободи для кожної суми квадратів
df_a = data.shape[0] - 1
df_b = data.shape[1] - 1
df_ab = df_a * df_b

# Обчислюємо середнє квадратичне для кожного фактора та взаємодії
ms_a = ss_a / df_a
ms_b = ss_b / df_b
ms_ab = ss_ab / df_ab

# Обчислюємо F-статистику для кожного фактора та взаємодії
f_a = ms_a / ms_ab
f_b = ms_b / ms_ab
f_ab = ms_ab / ms_ab

# Визначаємо критичне F-значення, використовуючи рівень значущості та ступені свободи
alpha = 0.1
crit_a = f.ppf(1 - alpha, df_a, df_ab)
crit_b = f.ppf(1 - alpha, df_b, df_ab)
crit_ab = f.ppf(1 - alpha, df_ab, df_ab)

# Порівнюємо F-статистику з критичним F-значенням, щоб визначити, чи фактор або взаємодія є значущими
if f_a > crit_a:
    print("Залежить від фактору A")
else:
    print("Не залежить від фактору A")

if f_b > crit_b:
    print("Залежить від фактору B")
else:
    print("Не залежить від фактору B")

if f_ab > crit_ab:
    print("Взаємодія між факторами A і B є значущою")
else:
    print("Взаємодія між факторами A і B не є значущою")
# fmt:on

mA = np.mean(data, axis=1)
deviation = data - mA.reshape(-1, 1)
xi = mA.reshape(-1, 1) + deviation
print(xi)