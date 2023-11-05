import numpy as np
from scipy.stats import f

# fmt:off
data = np.array([
    [3.836, 5.468, 5.784, 5.593], 
    [5.978, 6.131, 6.724, 7.264],
    [7.098, 6.954, 7.346, 8.136]
])
# fmt:on

rows, cols = data.shape
n = rows * cols

print(f"{rows=}, {cols=}, {n=}")

mean = np.mean(data)

mean_a = np.mean(data, axis=1)
mean_b = np.mean(data, axis=0)

print(f"mean={mean:.3f}\n{mean_a=}\n{mean_b=}")

sigma_a = (cols / n) * np.sum((mean_a - mean) ** 2)
sigma_b = (rows / n) * np.sum((mean_b - mean) ** 2)

# fmt:off
sigma_0 = (1 / n) * np.sum(((data - mean_a[:, np.newaxis] - mean_b[np.newaxis, :]) + mean) ** 2)
# fmt:on
sigma = (1 / n) * np.sum((data - mean) ** 2)

print(f"sigma_a={sigma_a:.3f}, sigma_b={sigma_b:.3f}, sigma_0={sigma_0:.3f}, sigma={sigma:.3f}")

alpha = 0.1

FA = sigma_a / sigma_0
F = f.ppf(1 - alpha, rows - 1, n - rows - cols + 1)
print(f"FA={FA:.3f}, F={F:.3f}")

if FA < F:
    print("Залежить від фактору A")
else:
    print("Не залежить від фактору A")

FB = sigma_b / sigma_0
F = f.ppf(1 - alpha, cols - 1, n - rows - cols + 1)
print(f"FB={FB:.3f}, F={F:.3f}")

if FB < F:
    print("Залежить від фактору B")
else:
    print("Не залежить від фактору B")

var = sigma_0 * n / (n- rows - cols + 1)
print(f"var={var:.3f}")
    
coff_a = sigma_a / sigma
coff_b = sigma_b / sigma

print(f"coff_a={coff_a:.3f}, coff_b={coff_b:.3f}")

# Уточнена модель

np.random.seed(0)
new_data = np.random.normal(0, np.sqrt(var), data.shape)

refined_model = mean_a[:, np.newaxis] + new_data

print(f"{data=}")
print(f"{refined_model=}")