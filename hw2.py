import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.figure()

# Читаем данные из дано и делаем выборку по значениям,
# которые входят в промежуток от 160 до 190
df = pd.read_csv("data.dat")
sample = df[(df["Height"] >= 160) & (df["Height"] <= 190)]
n = sample["Height"].size
print(f"Sample size: {n}")

# Задаем уровень доверительного интервала
alpha = 0.05

# Строим гистограмму роста
bin_edges = np.linspace(160, 190, 13)

ax1 = sample["Height"].plot.hist(density=True, bins=bin_edges, ec="black", grid=True)
ax1.set_xlabel("Height (cm)")
ax1.set_ylabel("Density")
ax1.set_title("Height histogram")

# Ищем выборочное среднее и выборочную дисперсию 
# с помощью методов mean() и var() соответственно
# Также ищем стандартную ошибку среднего
mean = sample["Height"].mean()
var = sample["Height"].var()
std = sample["Height"].std()
std_error = std / np.sqrt(n)

print(f"Sample mean: {mean}")
print(f"Sample variance: {var}")

# Ищем критическое значение t для уровня доверительного интервала 
# и числа степеней свободы
t_crit = stats.t.ppf(1 - alpha / 2, n-1)

# Ищем критические значения хи-квадрат распределения для уровня 
# доверительного инвервала и числа степеней свободы
chi2_lower = stats.chi2.ppf(alpha / 2, n-1)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, n-1)

# Ищем интервальную оценку для мат. ожидания
lower_mean = mean - t_crit * std_error
upper_mean = mean + t_crit * std_error
print(f"Interval estimation (mean): ({lower_mean}, {upper_mean})")

# Ищем интервальную оценку для дисперсии
lower_var = (n-1) * var / chi2_upper
upper_var = (n-1) * var / chi2_lower
print(f"Interval estimation (var): ({lower_var}, {upper_var})")

# Строим функцию плотности вероятности на том же графике, 
# что и гистограмма. Сначала вычисляем значения плотности вероятности
# для каждого бина, а после - наносим на график
x = np.linspace(160, 190, 1000)
pdf = stats.norm.pdf(x, loc=mean, scale=np.sqrt(var))
ax1.plot(x, pdf, color="red")

plt.savefig("hist.png")
plt.show()

# Вычисляем наблюдаемые частоты 
# и ищем ожидаемые частоты для каждого инвервала
observed, _ = np.histogram(sample["Height"], bins=bin_edges)
expected = n * np.diff(stats.norm.cdf(bin_edges, loc=mean, scale=std))

# Ищем статистику критерия хи-квадрат
chisq_statistic = np.sum((observed - expected)**2 / expected)
print(f"Chi-squared statistic: {chisq_statistic}")

# Ищем критическое значение хи-квадрат для заданного уровня значимости
df = len(observed) - 1
critical_value = stats.chi2.ppf(q=1-alpha, df=df)
print(f"Chi-squared critical value: {critical_value}")

# Определяем, принимаем ли мы или отвергаем гипотезу
if chisq_statistic > critical_value:
    result = "Reject"
else: 
    result = "Accept"

print(f"Chi-squared test: {result}")





