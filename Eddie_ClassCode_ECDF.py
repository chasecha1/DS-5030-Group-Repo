
#COMPUTE ECDF

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.random.random(200)
df = pd.DataFrame(data, columns=['Dummy Data'])


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

