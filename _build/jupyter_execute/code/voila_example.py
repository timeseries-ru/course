%matplotlib inline
import seaborn as sns
import numpy as np

Привет! Я пример интерактивной тетрадки, отображаемой с помощью `voila`!

from ipywidgets import interact

def plot_random(samples=50):
    np.random.seed(1)
    sns.distplot(np.random.normal(size=samples))
    
interact(plot_random, samples=(10, 500, 10));