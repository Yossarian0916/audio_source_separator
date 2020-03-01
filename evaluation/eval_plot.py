import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# agg backend is used to create plot as a .png file
mpl.use('agg')

eval_results = []

# create a figure instance
fig = plt.figure(1, figsize=(9, 6))
# create an axes instance
ax = fig.add_subplot(111)
# create the boxplot
bp = ax.boxplot(eval_results)
# save the figure
fig.savefig('name.png', bbox_inches='tight')
