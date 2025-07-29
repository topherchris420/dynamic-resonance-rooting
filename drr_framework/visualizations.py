import matplotlib.pyplot as plt

def plot_results(results, data):
    print("Plotting results...")
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title("Analysis Plot")
    plt.show()
