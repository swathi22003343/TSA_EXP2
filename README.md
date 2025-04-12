# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### Date:18/03/25
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib)

2. Load the dataset

3. Calculate the linear trend values using least square method

4. Calculate the polynomial trend values using least square method

5. End the program
### PROGRAM:
```
NAME : SWATHI D
REGISTER NUMBER :212222230154
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/content/MSFT(2000-2023).csv', parse_dates=['Date'], index_col='Date')
yearly_data = data['Adj Close'].resample('Y').mean().to_frame()
yearly_data.index = yearly_data.index.year
yearly_data.rename(columns={'Adj Close': 'Avg Adj Close'}, inplace=True)
def linear_trend(data):
    years = data.index.astype(int).tolist()
    values = data['Avg Adj Close'].tolist()
    X = [i - years[len(years) // 2] for i in years] 
    x2 = [i ** 2 for i in X]
    xy = [i * j for i, j in zip(X, values)]
    n = len(years)
    b = (n * sum(xy) - sum(values) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
    a = (sum(values) - b * sum(X)) / n
    trend = [a + b * X[i] for i in range(n)]
    return a, b, trend
def polynomial_trend(data, degree=2):
    years = data.index.astype(int).tolist()
    values = data['Avg Adj Close'].tolist()
    X = [i - years[len(years) // 2] for i in years]  
    

    matrix = [[sum(x**(i+j) for x in X) for j in range(degree + 1)] for i in range(degree + 1)]
    Y = [sum(values[k] * (X[k]**i) for k in range(len(X))) for i in range(degree + 1)]

    coeffs = np.linalg.solve(np.array(matrix), np.array(Y))
    trend = [sum(coeffs[j] * (X[i]**j) for j in range(degree + 1)) for i in range(len(X))]
    
    return coeffs, trend

lin_a, lin_b, yearly_linear_trend = linear_trend(yearly_data)
yearly_data['Linear Trend'] = yearly_linear_trend

poly_coeff, yearly_poly_trend = polynomial_trend(yearly_data, degree=2)
yearly_data['Polynomial Trend'] = yearly_poly_trend

yearly_data.set_index(pd.Index(yearly_data.index.astype(str)), inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(yearly_data.index, yearly_data['Avg Adj Close'], 'bo-', label="Stock market price")
plt.plot(yearly_data.index, yearly_data['Linear Trend'], 'k--', label="Linear Trend")
plt.xlabel('Year')
plt.ylabel('Avg Adjusted Close Price')
plt.title('Linear Trend Estimation (MSFT 2000-2023)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(yearly_data.index, yearly_data['Avg Adj Close'], 'bo-', label="Stock market price")
plt.plot(yearly_data.index, yearly_data['Polynomial Trend'], 'g-', label="Polynomial Trend")
plt.xlabel('Year')
plt.ylabel('Avg Adjusted Close Price')
plt.title('Polynomial Trend Estimation (MSFT 2000-2023)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Linear Trend Equation: y = {lin_a:.2f} + {lin_b:.2f}x")
print(f"Polynomial Trend Equation: y = {poly_coeff[0]:.2f} + {poly_coeff[1]:.2f}x + {poly_coeff[2]:.2f}x²")
```
### OUTPUT
### A - LINEAR TREND ESTIMATION
![image](https://github.com/user-attachments/assets/0f875714-5e41-435f-b095-5e806aa8a408)


### B- POLYNOMIAL TREND ESTIMATION
![image](https://github.com/user-attachments/assets/8a4bce34-d25b-43c1-8a44-12e6aceef0bf)


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
