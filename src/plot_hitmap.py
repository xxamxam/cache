from joblib import Parallel, delayed
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_hit_map(requests, cache_prob, K, N, show_plot = True):
    del_cache = delayed(cache_prob)
    betas_len = 6
    gammas_len = 10
    gammas = np.logspace( np.log10(0.02), np.log10(0.00001), gammas_len)
    betas = np.logspace(np.log10(0.2), np.log10(0.001), betas_len)
    grid = np.meshgrid(gammas, betas)
    grid_misses = Parallel(n_jobs=8)( del_cache(requests, K, N, gammas[i], betas[j]) for i in range(gammas_len) for j in range(betas_len)  )
    grid_misses = np.array([grid_misses[i][0] for i in range(len(grid_misses))])
    grid_misses = grid_misses.reshape(gammas_len, betas_len).T

    dataframe = pd.DataFrame(data = grid_misses, index = [f"{num:.2e}" for num in betas], columns= [f"{num:.2e}" for num in gammas])
   
    if show_plot:
        sns.heatmap(dataframe * 100, annot = True)
        plt.xlabel("gamma")
        plt.ylabel('beta')
        plt.show()
    ind = np.unravel_index(dataframe.values.argmin(), dataframe.values.shape)
    print(f"best res: {dataframe.iloc[ind]}\n \
            with paramenters: gamma={dataframe.columns[ind[1]]}, beta = {dataframe.index[ind[0]]}") 
    return dataframe

def plot_hit_map_mul(requests, cache_prob_mul, K, N, show_plot = True):
    del_cache = delayed(cache_prob_mul)
    betas_len = 8
    gammas_len = 12
    gammas = np.logspace( np.log10(0.2), np.log10(0.000001), gammas_len)
    betas = np.logspace(np.log10(1.01), np.log10(1.3), betas_len)
    grid = np.meshgrid(gammas, betas)
    grid_misses = Parallel(n_jobs=9)( del_cache(requests, K, N, gammas[i], betas[j]) for i in range(gammas_len) for j in range(betas_len)  )
    grid_misses = np.array([grid_misses[i][0] for i in range(len(grid_misses))])
    grid_misses = grid_misses.reshape(gammas_len, betas_len).T

    probs = np.mean([grid_misses[i][2] for i in range(len(grid_misses))])
    print(f"mean alive prob: {probs}")
    dataframe = pd.DataFrame(data = grid_misses, index = [f"{num:.2e}" for num in betas], columns= [f"{num:.2e}" for num in gammas])
    if show_plot:
        sns.heatmap(dataframe * 100, annot = True)
        plt.xlabel("gamma")
        plt.ylabel('beta')
        plt.show()

    ind = np.unravel_index(dataframe.values.argmin(), dataframe.values.shape)
    print(f"best res: {dataframe.iloc[ind]}\n \
            with paramenters: gamma={dataframe.columns[ind[1]]}, beta = {dataframe.index[ind[0]]}") 
    
    return dataframe