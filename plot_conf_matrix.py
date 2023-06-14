import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import JH_utile


# DS = np.array([89.22, 88.04, 89.11, 88.64, 88.94])
#
# kernel_3 = np.array([90.13, 90.59, 91.51, 90.43, 90.26])
#
# # kernel_5 = np.array([91.05, 91.97, 91.57, 92.43, 91.84])
#
# mean, std = JH_utile.meanstd2(kernel_5)
# print()



True_label = ["Baseline", "Loaded", "Rapid"]

predicted_class = ["Baseline", "Loaded", "Rapid"]

conf_matrix = np.array([[0.92, 0.06, 0.03],
                        [0.10, 0.81, 0.10],
                        [0.02, 0.02, 0.96],
                        ])

fig, ax = plt.subplots()

im, cbar = JH_utile.heatmap(conf_matrix, True_label, predicted_class, ax=ax,
                   cmap="jet", cbarlabel="data range")
texts = JH_utile.annotate_heatmap(im, valfmt="{x:.2f}", threshold=0.5)

fig.tight_layout()
plt.show()

fig.savefig('data_plot')