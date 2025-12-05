
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

measured_data = np.array([
[29.77943442, -95.66190641, 32.94, 51.42], [29.75410235, -95.66211382, 23.05, 75.24], [29.76237765, -95.66773468, 32.41, 56.72], 
[29.76055141, -95.67394637, 13.83, 83.46], [29.76249313, -95.67344804, 14.21, 85.83], [29.77577207, -95.66802898, 33.12, 56.84], 
[29.75590438, -95.67776561, 36.65, 44.89], [29.77750692, -95.66985163, 52.84, 19.36], [29.77658979, -95.66473542, 44.28, 24.9], 
[29.7673653, -95.67088005, 37.67, 35.43], [29.75178114, -95.67246514, 50.72, 17.95], [29.77398171, -95.65376042, 48.92, 30.35], 
[29.76149727, -95.66803651, 10.0, 92.1], [29.76362807, -95.65138231, 55.0, 15.24], [29.75473965, -95.67451084, 19.66, 79.14], 
[29.76437982, -95.66116455, 40.53, 43.19], [29.77767764, -95.66584378, 18.09, 83.32], [29.7632549, -95.65774477, 39.04, 59.06], 
[29.77695812, -95.67218302, 19.37, 78.33], [29.75597252, -95.65171902, 32.2, 59.05], [29.77631455, -95.67935152, 42.37, 34.84], 
[29.76445596, -95.67086871, 31.14, 61.47], [29.77738933, -95.66389744, 29.49, 58.6], [29.77404585, -95.66705273, 21.51, 70.3], 
[29.76910267, -95.67093529, 25.15, 74.54], [29.7630618, -95.6514662, 18.96, 75.27], [29.76594878, -95.67282026, 35.7, 49.41], 
[29.77418253, -95.66330937, 52.32, 22.92], [29.77419508, -95.66232536, 28.25, 66.2], [29.77130697, -95.67520484, 25.5, 71.75], 
[29.75995226, -95.67798914, 51.22, 17.52], [29.76947443, -95.66534001, 12.94, 87.35], [29.75555676, -95.67177653, 35.9, 49.8], 
[29.76422119, -95.67696728, 21.8, 66.44], [29.7641756, -95.65068453, 39.49, 41.51], [29.76769784, -95.67015136, 21.32, 67.43], 
[29.7772967, -95.67309607, 15.1, 81.6], [29.77943843, -95.66462662, 36.59, 58.26], [29.76677695, -95.67393677, 43.95, 29.22], 
[29.75620745, -95.670986, 43.2, 30.7], [29.75185289, -95.65649605, 31.46, 63.9], [29.75294749, -95.65541075, 10.0, 94.33], 
[29.77335045, -95.67090506, 39.53, 50.24], [29.77892573, -95.6763542, 23.97, 72.38], [29.76718689, -95.67781587, 22.23, 74.42], 
[29.76988899, -95.6799631, 38.22, 56.52], [29.76712933, -95.65185673, 46.05, 23.95], [29.77606051, -95.65007221, 41.67, 35.99], 
[29.77707953, -95.67031863, 45.63, 36.46], [29.77804853, -95.65017023, 40.28, 38.48], [29.7707118, -95.67812837, 54.88, 21.16], 
[29.75408449, -95.66749701, 43.03, 38.99], [29.76704983, -95.67448489, 53.12, 16.61], [29.76954754, -95.65884739, 34.06, 52.86], 
[29.76188235, -95.65603242, 20.08, 81.87], [29.77370643, -95.65893095, 24.44, 67.54], [29.76222937, -95.65642867, 21.15, 73.94], 
[29.76629792, -95.67651989, 55.0, 16.0], [29.75191262, -95.65654328, 14.61, 82.16], [29.77651738, -95.65273326, 38.21, 34.05]
])

latitudes = measured_data[:, 0]
longitudes = measured_data[:, 1]
y = measured_data[:, 2] 
x = measured_data[:, 3] 
sumY = sum(y) 
sumX = sum(x) 
num = len(y) 
meanY = sumY/num 
meanX = sumX/num


Sycomp = 0 
Sxcomp = 0

for value in x: 
    Sxcomp += (value - meanX)**2
for value in y: 
    Sycomp += (value - meanY)**2


Sx = ((Sxcomp)/(num - 1))**(1/2)
Sy = ((Sycomp)/(num - 1))**(1/2) 

def calculate_r_squared(y_true, y_predicted): 
    ss_total = ((y_true - np.mean(y_true)) ** 2).sum()
    ss_residual = ((y_true - y_predicted) ** 2).sum()

    if ss_total == 0: 
        return 1.0 if ss_residual == 0 else 0.0 
    r_squared = 1 - (ss_residual)/(ss_total)
    return(r_squared) 

coefficients_linear = np.polyfit(x,y,1) 
y_predicted_linear = np.polyval(coefficients_linear, x) 
r_squared_linear = calculate_r_squared(y, y_predicted_linear)
slope_linear = coefficients_linear[0] 
intercept_linear = coefficients_linear[1] 

r_linear = slope_linear * (Sx/Sy)
print("**Linear**")
print("y = " + str(slope_linear) + "x" + " + " + str(intercept_linear))
print("r^2: " + str(r_squared_linear)) 
linear_model = np.poly1d(coefficients_linear)

coefficients_quadratic = np.polyfit(x,y,2) 
y_predicted_quadratic = np.polyval(coefficients_quadratic, x) 
r_squared_quadratic = calculate_r_squared(y, y_predicted_quadratic)
a_quadratic = coefficients_quadratic[0] 
b_quadratic = coefficients_quadratic[1] 
intercept_quadratic = coefficients_quadratic[2] 
print("      ")
print("*Quadratic**")
print("y = " + str(a_quadratic) + "x^2" + " + " + str(b_quadratic) + "x" + " + " + str(intercept_quadratic))
print("r^2: " + str(r_squared_quadratic))
quadratic_model = np.poly1d(coefficients_quadratic)

coefficients_cubic = np.polyfit(x,y,3)
y_predicted_cubic = np.polyval(coefficients_cubic, x) 
r_squared_cubic = calculate_r_squared(y, y_predicted_cubic) 
a_cubic = coefficients_cubic[0] 
b_cubic = coefficients_cubic[1] 
c_cubic = coefficients_cubic[2] 
intercept_cubic = coefficients_cubic[3]
print("      ")
print("*Cubic**")
print("y = " + str(a_cubic) + "x^3" + " + " + str(b_cubic) + "x^2" + " + " + str(c_cubic) + "x " + " + " + str(intercept_cubic))
print("r^2: " + str(r_squared_cubic))
cubic_model = np.poly1d(coefficients_cubic)

def logarithmic_func(x, a, b):
    return a * np.log(x) + b

popt_log, pcov_log = curve_fit(logarithmic_func, x, y, p0=[1, 1])
a_log, b_log = popt_log
y_pred_log = logarithmic_func(x, a_log, b_log)
r_squared_log = calculate_r_squared(y, y_pred_log)
print("      ")
print("**Logarithmic**")
print(f"y = {a_log:.4f} * ln(x) + {b_log:.4f}")
print("r^2: " + str(r_squared_log))

def exponential_func(x, a, b):
    return a * np.exp(b * x)


popt_exp, pcov_exp = curve_fit(exponential_func, x, y, p0=[1, 0.01])
a_exp, b_exp = popt_exp
y_pred_exp = exponential_func(x, a_exp, b_exp)
r_squared_exp = calculate_r_squared(y, y_pred_exp)
print("      ")
print("**Exponential**")
print(f"y = {a_exp:.4f} * exp({b_exp:.4f} * x)")
print("r^2: " + str(r_squared_exp))

if(r_squared_linear > r_squared_quadratic and r_squared_linear > r_squared_cubic and r_squared_linear > r_squared_log and r_squared_linear > r_squared_exp): 
    recommended_model = "Linear" 
elif(r_squared_quadratic > r_squared_linear and r_squared_quadratic > r_squared_cubic and r_squared_quadratic > r_squared_log and r_squared_quadratic > r_squared_exp): 
    recommended_model = "Quadratic"
elif(r_squared_cubic > r_squared_quadratic and r_squared_quadratic < r_squared_cubic and r_squared_cubic > r_squared_log and r_squared_cubic > r_squared_exp): 
    recommended_model = "Cubic"
elif(r_squared_log > r_squared_quadratic and r_squared_log > r_squared_cubic and r_squared_linear < r_squared_log and r_squared_log > r_squared_exp): 
    recommended_model = "Logarithmic"
elif(r_squared_exp > r_squared_quadratic and r_squared_exp > r_squared_cubic and r_squared_exp > r_squared_log and r_squared_linear < r_squared_exp): 
    recommended_model = "Exponential"
else: 
    recommended_model = "could not be determined."
print("      ")
print("where y = predicted moisture level and x = elevation at that point.")
print("      ")
print("Recommended model (highest r^2 value): " + recommended_model)

exponential_model = lambda x_val: exponential_func(x_val, a_exp, b_exp)
logarithmic_model_curve_fit = lambda x_val: logarithmic_func(x_val, a_log, b_log)
x_fit = np.linspace(min(x), max(x), 100)

plt.figure(figsize=(10,6))

plt.scatter(x, y, color='red', label='Measured Data')
plt.plot(x_fit, linear_model(x_fit), 
         label = "Linear Fit",
         color ='blue', linestyle='--') 

plt.plot(x_fit, quadratic_model(x_fit), 
         label = "Quadratic Fit", 
         color ='green', linestyle='--')

plt.plot(x_fit, cubic_model(x_fit), 
         label = "Cubic Fit", 
         color ='orange', linestyle='--')  

plt.plot(x_fit, logarithmic_model_curve_fit(x_fit), 
         label = "Logarithmic Fit", 
         color ='black', linestyle='--')

plt.plot(x_fit, exponential_model(x_fit), 
         label = "Exponential Fit", 
         color ='red', linestyle='--')


plt.title('Regression on Measured Drone Data')
plt.xlabel('Elevation (Auxillary Variable)') 
plt.ylabel('Moisture Values')
plt.legend() 
plt.grid(True) 
plt.show()


residuals_linear_list = []

for i in range(len(x)):
    y_actual = y[i]
    y_predicted = linear_model(x[i])
    
    residual = y_actual - y_predicted
    residuals_linear_list.append(residual)

residuals_linear_array = np.array(residuals_linear_list)
spatial_residuals_linear = np.column_stack((latitudes, longitudes, residuals_linear_array))


residuals_quadratic_list = []

for i in range(len(x)):
    y_actual = y[i]
    y_predicted = quadratic_model(x[i])
    
    residual = y_actual - y_predicted
    residuals_quadratic_list.append(residual)

residuals_quadratic_array = np.array(residuals_quadratic_list)
spatial_residuals_quadratic = np.column_stack((latitudes, longitudes, residuals_quadratic_array))


residuals_cubic_list = []

for i in range(len(x)):
    y_actual = y[i]
    y_predicted = cubic_model(x[i])
    
    residual = y_actual - y_predicted
    residuals_cubic_list.append(residual)

residuals_cubic_array = np.array(residuals_cubic_list)
spatial_residuals_cubic = np.column_stack((latitudes, longitudes, residuals_cubic_array))

y_predicted_log = logarithmic_model_curve_fit(x) 
residuals_log = y - y_predicted_log 
spatial_residuals_log = np.column_stack((latitudes, longitudes, residuals_log))

y_predicted_exp = exponential_model(x) 
residuals_exp = y - y_predicted_exp
spatial_residuals_exp = np.column_stack((latitudes, longitudes, residuals_exp))