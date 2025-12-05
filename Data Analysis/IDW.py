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

#data collected by the drone, we can copy/paste this in from the text file

all_latitudes = [lat for lat, lon, moisture in measured_data] 
#list of latitudes, first element in each data bunch

all_longnitudes = [lon for lat, lon, moisture in measured_data ]
#list of longnitudes, second element in each data bunch

all_moisture = [moisture for lat, lon, moisture in measured_data]
#list of moisture readings, third element in each data bunch

points_list = [[lon,lat] for lat, lon, moisture in measured_data]

#list of points, results in the form [longnitude, latitude] 

grid_points = np.array(points_list) 

'''
converts the list of points into a 2d array with 11 rows and 2 columns,
first column is longnitude and second is latitude, 
11 rows because there were 11 points measured in measured_data
'''
power_parameter = 2

min_moisture = min(all_moisture) #smallest moisture value
max_moisture = max(all_moisture) #largest moisture value

grid_resolution = 75 
#how many pixels there will be on the generated heatmap


long_minimum  = min(all_longnitudes) #smallest longnitude
long_maximum = max(all_longnitudes) #largest longnitude

lat_minimum = min(all_latitudes) #smallest latitude
lat_maximum = max(all_latitudes) #largest latitude

xi = np.linspace(long_minimum, long_maximum, grid_resolution)
'''
equally space out points between the minimum and maximum longnitudes, for a total of grid_resolution x points, 
xi is holding 50 longnitude values as a 1D array
'''

yi = np.linspace(lat_minimum, lat_maximum, grid_resolution) 

'''
equally space out points between the minimum and maximum latitudes, for a total of grid_resolution y points
yi is holding 50 latitude values as a 1D array
'''

XI, YI = np.meshgrid(xi, yi)

'''
turns xi,yi (sort of) into a grid, XI and YI are BOTH 50x50 matrices 
XI (Longitude Matrix):
- Formed by repeating the 1D 'xi' array 50 times down the rows.
- The values are constant vertically: all elements in any single COLUMN are the same.

YI (Latitude Matrix):
- Formed by repeating the 1D 'yi' array 50 times across the columns.
- The values are constant horizontally: all elements in any single ROW are the same.

 now the coordinates of the grid are defined, 
 i.e the coordinates at (6,7) would be (XI[6,7], YI[6,7]) 
 [a,b] --> ath row, bth column
 '''

predicted_points = np.vstack([XI.flatten(), YI.flatten()]).T 

'''
the .flatten() function compresses the XI and YI matrices into 1D arrays with 2500 elements each, the .T means transpose, 
converting the grid from 2 rows and 2500 columns to 2500 rows and 2 columns, one column for XI and the other for YI
'''

distances = cdist(predicted_points, grid_points, metric='euclidean')
'''
distances becomes a 2500 x 11 array (11 because there are 11 data entries in measured_data), cdist calculates the distance between 
each of the 2500 grid points and each of the 11 measured points
each column represents the 2500 distances for ONE measured point
'''

distances[distances == 0] = 1e-14 
'''
if the grid point and measured point are the exact same point we would have to divide by 0
so in this scenario we set the distance to a very very small amount close to 0
'''
weights = 1/(distances**power_parameter)
'''
weights becomes an array with 2500 rows and 11 columns, 
each column represents one of the 11 meausred points and its influence on the 2500 points on the grid
'''
moisture_values = np.array(all_moisture)
#converting moisture_values into an np array to avoid type error, 1D array with 11 values

interpolated = np.sum(weights * moisture_values, axis = 1)/np.sum(weights, axis = 1)

'''
calculating the weighted average of each point, this represents the predicted value for every grid point
axis = 1 means we are summing all elements on each row
'''

interpolated_matrix = interpolated.reshape(grid_resolution, grid_resolution) 

'''
converting interpolated from a 2500 element long 1D array into a 50x50 array
the positions on this array correspond to the IDW interpolated moisture values 
on the actual generated grid.

'''

plt.pcolormesh( 
    XI, #represents longitudes or x coordinates 
    YI, #represents latitudes or Y coordinates 
    interpolated_matrix, #represents the predicted moisture values on the grid, putting these values on the grid like a blanket
    cmap = 'plasma_r', #color scheme
    vmin = min_moisture, #minimum moisture on the color scale 
    vmax = max_moisture, #maximum moisture on the color scale  

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
