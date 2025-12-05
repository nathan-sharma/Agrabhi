#Coded by Nathan

import math
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 

measured_data = [
[29.7651805617053, -95.6726225674295, 50],
[29.7646480139456, -95.66770566509483, 50],
[29.766770356606497, -95.66585618777385, 50 ],
[29.769341931161193, -95.65931057664636, 48.00],
[29.771679101955915, -95.66245897483506, 30],
[29.774215269001424, -95.65850751876573, 21],
[29.7793024769016, -95.6795047316388, 20],
[29.761418595973392, -95.67243106176178, 25],
[29.756442237115426, -95.66944446965238, 15],
[29.76133496173184, -95.66303774786935, 17],
[29.752929364436035, -95.67228654924037, 19.00]
]

all_latitudes = [lat for lat, lon, moisture in measured_data] 

all_longnitudes = [lon for lat, lon, moisture in measured_data ]


all_moisture = [moisture for lat, lon, moisture in measured_data]


points_list = [[lon,lat] for lat, lon, moisture in measured_data]



grid_points = np.array(points_list) 


power_parameter = 2

min_moisture = min(all_moisture) 
max_moisture = max(all_moisture) 

grid_resolution = 75 



long_minimum  = min(all_longnitudes)
long_maximum = max(all_longnitudes) 

lat_minimum = min(all_latitudes) 
lat_maximum = max(all_latitudes) 

xi = np.linspace(long_minimum, long_maximum, grid_resolution)


yi = np.linspace(lat_minimum, lat_maximum, grid_resolution) 



XI, YI = np.meshgrid(xi, yi)



predicted_points = np.vstack([XI.flatten(), YI.flatten()]).T 



distances = cdist(predicted_points, grid_points, metric='euclidean')


distances[distances == 0] = 1e-14 

weights = 1/(distances**power_parameter)

moisture_values = np.array(all_moisture)


interpolated = np.sum(weights * moisture_values, axis = 1)/np.sum(weights, axis = 1)


interpolated_matrix = interpolated.reshape(grid_resolution, grid_resolution) 


plt.pcolormesh( 
    XI,
    YI,  
    interpolated_matrix, 
    cmap = 'plasma_r', 
    vmin = min_moisture, 
    vmax = max_moisture, 

)


plt.scatter ( 
    all_longnitudes, 
    all_latitudes, 
    marker = 's', 
    c = all_moisture, 
    cmap = 'plasma_r',
    s = 200, 
    vmin = min_moisture, 
    vmax = max_moisture)

plt.colorbar(label = 'Moisture') 

plt.xlabel('Longitude') 
plt.ylabel('Latitude') 
plt.title('IDW Heatmap')
plt.show() 
