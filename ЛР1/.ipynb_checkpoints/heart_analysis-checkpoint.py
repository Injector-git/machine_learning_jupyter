import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('heart.csv')

#17 первых строк
print("Первые 17 строк:")
print(df.head(17))
print("\n" + "="*60 + "\n")


#Cколько строк и столбцов в наборе данных?
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")
print("\n" + "="*60 + "\n")

#сводная информацию по датафрейму
print("Сводная информация:")
print(df.info())
print("\n")
print("Статистическое описание:")
print(df.describe())
print("\n" + "="*60 + "\n")

#Набор данных на наличие отсутствующих значений и дубликатов строк
print("Отсутствующие значения:")
print(df.isnull().sum())
print("\nКоличество дубликатов строк:", df.duplicated().sum())

# Обработка: удалить дубликаты (если есть)
df = df.drop_duplicates()
# В данном датасете пропусков нет, но если бы были, можно было бы заполнить медианой или удалить
# Например: df = df.fillna(df.median())
print("\nПосле удаления дубликатов строк осталось:", df.shape[0])
print("\n" + "="*60 + "\n")

#Два датафрейма, поместив в них отдельно здоровых и больных пациентов
df_healthy = df[df['target'] == 0]  # здоровые (нет заболевания)
df_sick = df[df['target'] == 1]     # больные (есть заболевание)

print(f"Здоровых пациентов: {len(df_healthy)}")
print(f"Больных пациентов: {len(df_sick)}")
print("\n" + "="*60 + "\n")

#Каков средний возраст пациентов в той и другой группе?
print(f"Средний возраст здоровых: {df_healthy['age'].mean():.2f} лет")
print(f"Средний возраст больных: {df_sick['age'].mean():.2f} лет")
print("\n" + "="*60 + "\n")

#Как влияет максимальная достигнутая частота сердечных сокращений на наличие заболевания?
print("Средняя максимальная ЧСС у здоровых:", df_healthy['thalach'].mean())
print("Средняя максимальная ЧСС у больных:", df_sick['thalach'].mean())
# Чем выше ЧСС, тем реже заболевание (у больных в среднем ЧСС ниже)
print("\n" + "="*60 + "\n")

#Как влияет уровень сахара в крови на наличие заболевания?
# fbs: 1 - уровень сахара > 120 мг/дл, 0 - иначе
fbs_healthy = df_healthy['fbs'].value_counts(normalize=True)
fbs_sick = df_sick['fbs'].value_counts(normalize=True)
print("Доля повышенного сахара (fbs=1) среди здоровых:", fbs_healthy.get(1, 0))
print("Доля повышенного сахара (fbs=1) среди больных:", fbs_sick.get(1, 0))
print("Вывод: повышенный сахар слабо коррелирует с заболеванием (разница небольшая).")
print("\n" + "="*60 + "\n")

#Как влияет уровень холестерина на наличие заболевания?
print("Средний холестерин у здоровых:", df_healthy['chol'].mean())
print("Средний холестерин у больных:", df_sick['chol'].mean())
print("У больных холестерин в среднем выше.")
print("\n" + "="*60 + "\n")

#Влияет ли пол пациента на наличие заболевания?
#sex: 1 - мужской, 0 - женский
male_healthy = df_healthy[df_healthy['sex'] == 1].shape[0] / len(df_healthy)
male_sick = df_sick[df_sick['sex'] == 1].shape[0] / len(df_sick)
print(f"Доля мужчин среди здоровых: {male_healthy:.2%}")
print(f"Доля мужчин среди больных: {male_sick:.2%}")
print("Мужчины болеют чаще (большая доля среди больных).")
print("\n" + "="*60 + "\n")

#TOP-5 самых старых и TOP-5 самых молодых пациентов с болезнью сердца
oldest_sick = df_sick.nlargest(5, 'age')[['age', 'sex', 'trestbps', 'chol', 'target']]
youngest_sick = df_sick.nsmallest(5, 'age')[['age', 'sex', 'trestbps', 'chol', 'target']]

print("TOP-5 самых старых пациентов с болезнью сердца:")
print(oldest_sick)
print("\nTOP-5 самых молодых пациентов с болезнью сердца:")
print(youngest_sick)
print("\n" + "="*60 + "\n")

#Преобразование значения числового столбца trestbps в числа с плавающей запятой
df['trestbps'] = df['trestbps'].astype(float)
print("Тип данных после преобразования trestbps:", df['trestbps'].dtype)
print("\n" + "="*60 + "\n")

#Вычисление для этого столбца среднее значение, моду, медиану, дисперсию и среднеквадратичное отклонение
mean_val = df['trestbps'].mean()
mode_val = df['trestbps'].mode()[0]  # первая мода
median_val = df['trestbps'].median()
variance_val = df['trestbps'].var()
std_val = df['trestbps'].std()

print(f"Среднее значение trestbps: {mean_val:.2f}")
print(f"Мода trestbps: {mode_val:.2f}")
print(f"Медиана trestbps: {median_val:.2f}")
print(f"Дисперсия trestbps: {variance_val:.2f}")
print(f"Среднеквадратичное отклонение trestbps: {std_val:.2f}")
