from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

stocks = [
    [0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2, 2.5, 3],
    [1.59, 1.93, 1.95, 1.99, 2.22, 2.77, 2.80, 2.94, 2.98, 3.],
    [1.59, 1.93, 1.95, 1.99, 2.22, 2.77, 2.80, 2.94, 2.98, 3.]
    ]

stock_capacities = [
    [400, 400, 400, 400, 400, 400, 400, 400, 400],
    [540, 54, 54, 180, 270, 252, 180, 117, 27, 90],
    [200, 20, 20, 60, 110, 100, 65, 45, 5, 40]
]

stock_names = ['stock_uniform', 'stock_nonuniform1', 'stock_nonuniform2']


colors = [cm.jet(np.linspace(0, 1, len(stock))) for stock in stocks]

for stock, stock_capacity, color, name in zip(stocks, stock_capacities, colors, stock_names):
    # Creata a mew figure with custom dimensions
    plt.figure(figsize=(10, 5))

    plt.bar([str(s) for s in stock], stock_capacity, edgecolor='black', linewidth=0.5, color=color,  width=0.5)
    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(np.arange(0, max(stock_capacity) + 100, 100), fontsize=16)
    plt.xlabel('Stock length (m)', fontsize=16)
    plt.ylabel('Stock capacity', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
