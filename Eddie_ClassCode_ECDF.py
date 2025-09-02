# Eddie Anderson ECDF Class Exercise
# Author: Edward Anderson (eca4zm)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.array([1,2,2,3,3,3,4,4,4,4, 5])
df = pd.DataFrame(data, columns=['Dummy Data'])

# Empirical Cumulative Distribution Function Work
Z = np.sort(df['Dummy Data'].unique())
X = df['Dummy Data'].to_numpy().reshape(-1,1)
comparison = X <= Z
ecdf = np.mean(comparison, axis=0)
plt.plot(Z, ecdf)
plt.title('Empirical CDF with dummy data')
plt.ylabel('Proportion')
plt.xlabel('Dummy Data')


# Median Work
QUANTILE = .5
squared_distance = (ecdf - QUANTILE) ** 2 # Looking at the distance from the input quantile
min_index = np.where(squared_distance == squared_distance.min()) # getting the index of that quantile
print(Z[min_index])
median = np.mean(Z[min_index])
print('Median:', median)




def ecdf(x, plot = True):
    # Compute ecdf function:
    Z = np.sort(x.unique()) # Extract and sort unique values for x
    compare = x.to_numpy().reshape(-1,1) <= Z.reshape(1,-1) # Compare x and Z values    
    ecdf = np.mean(compare,axis=0) # Average over x indices for each z
    
    if plot:
        # Plot the ecdf:
        title_str = x.name
        plt.plot(Z,ecdf)
        plt.title(f'Empirical CDF: {title_str}')
        plt.ylabel('Proportion')
        plt.xlabel(title_str)

    return ecdf, Z
F_hat, grid = ecdf(df['Dummy Data'])



def compute_median(F_hat, grid, quantile):
    dist = (F_hat - quantile) ** 2 # Compute squared distance of F_hat from .5
    meds = np.where( dist == dist.min() ) # Find the indices closest to 1/2
    print(grid[meds])
    median = np.mean(grid[meds])
    return median

print(compute_median(F_hat, grid, .5))

