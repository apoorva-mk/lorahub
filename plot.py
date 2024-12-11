import matplotlib.pyplot as plt
from tueplots import axes, bundles
from tueplots import figsizes, fonts

# Increase the resolution of all the plots below
plt.rcParams.update({"figure.dpi": 150})

# Data from the table
tasks = [
    "Salient Translation Error Detection",
    "Logical Deduction Five Objects",
    "Movie Recommendation",
    "Geometric Shapes",
    "Hyperbaton",
    "Reasoning About Colored Objects",
    "Navigate",
    "Object Counting"
]

loRA_values = [3, 5, 7, 10, 15, 20]  # Number of LoRAs

# Accuracies for each task (corresponding to the number of LoRAs)
accuracies = {
    "Salient Translation Error Detection": [37.333, 37.333, 37.333, 37.333, 37.333, 40.000],
    "Logical Deduction Five Objects": [21.333, 21.333, 21.333, 21.333, 21.333, 23.333],
    "Movie Recommendation": [62.667, 62.667, 62.667, 62.667, 62.667, 60.667],
    "Geometric Shapes": [6.667, 6.667, 6.667, 6.667, 6.667, 7.333],
    "Hyperbaton": [7.333, 6.667, 6.667, 6.667, 6.000, 14.667],
    "Reasoning About Colored Objects": [32.000, 32.000, 32.000, 32.000, 32.000, 34.667],
    "Navigate": [47.333, 47.333, 47.333, 47.333, 47.333, 46.000],
    "Object Counting": [34.667, 34.667, 34.667, 34.667, 34.667, 35.333]
}
plt.rcParams.update(fonts.jmlr2001_tex(family="serif"))
with plt.rc_context({**bundles.icml2022(), **axes.lines()}):
    fig, ax = plt.subplots()
    ax.plot([1.0, 2.0], [3.0, 4.0], label="p(x)")
    ax.set_title("Title")
    ax.set_xlabel("xlabel")
    ax.set_ylabel("ylabel")
    plt.grid()
    plt.legend()
    plt.show()

# Save the plot as an image file (e.g., 'task_accuracies.png')
# plt.savefig('task_accuracies.png', dpi=300)  # You can specify the desired DPI

