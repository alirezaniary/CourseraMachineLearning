def plotData(func, x, y, title, xLabel, yLabel, marker='x', color='red'):
    import matplotlib.pyplot as plt
    if func == 'scatter':
        plt.scatter(x,y,marker=marker,c=color)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
    elif func == 'plot':
        plt.plot(x, y, color=color)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
    pass
