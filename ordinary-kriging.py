import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 
from pykrige.ok import OrdinaryKriging 
from regression import spatial_residuals_cubic, coefficients_cubic, x, y 


data = spatial_residuals_cubic 
lat, lon, z = data[:, 0], data[:, 1], data[:, 2] # Extract Y, X, and Residual (Z)


num_points = 200
grid_x = np.linspace(lon.min(), lon.max(), num_points)
grid_y = np.linspace(lat.min(), lat.max(), num_points)




OK_res = OrdinaryKriging( 
    lon, lat, z, variogram_model='spherical',  
    verbose=False, enable_plotting=False
)
z_kriged, _ = OK_res.execute("grid", grid_x, grid_y)

print(f"Sill: {OK_res.variogram_model_parameters[0]:.4f}")
print(f"Range: {OK_res.variogram_model_parameters[1]:.4f}")
print(f"Nugget: {OK_res.variogram_model_parameters[2]:.4f}")

def cubic_trend(x_val):
    a, b, c, d = coefficients_cubic
    return a * x_val**3 + b * x_val**2 + c * x_val + d

cubic_trend_1D = cubic_trend(x)
trend_OK = OrdinaryKriging(
    lon, lat, cubic_trend_1D, 
    variogram_model='spherical', nlags=5
)
cubic_trend_kriged, _ = trend_OK.execute("grid", grid_x, grid_y)

final_map = cubic_trend_kriged + z_kriged

map_min, map_max = final_map.min(), final_map.max()

plt.figure(figsize=(10, 8))


plt.contourf(
    grid_x, grid_y, final_map.T, 
    levels=100, cmap='YlGnBu', 
    vmin=map_min, vmax=map_max
)
plt.colorbar(label='Final Predicted Moisture Value')


plt.scatter(lon, lat, c=y, marker='o', edgecolors='black', label='Original Moisture Data')

plt.title('Final Regression Kriging Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
