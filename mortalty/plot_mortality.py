from PIL import Image
import numpy as np
from utils.tools import print_results, plot_example_box
import os
path = './results/'
explainers = [
            "occlusion",
            "augmented_occlusion",
            "integrated_gradients",
            "deep_lift",
            "lime",
            "fit",
            "retain",
            "dyna_mask",
            "extremal_mask",
            "gate_mask",
]
names = {
    "occlusion":"FO",
    "augmented_occlusion":"AFO",
    "integrated_gradients":"IG",
    "gradient_shap":"GradShap",
    "deep_lift":"DeepLift",
    "lime":"LIME",
    "fit":"FIT",
    "retain":"RETAIN",
    "dyna_mask":"Dynamask",
    "extremal_mask":"Extrmask",
    "gate_mask":"ContraLSP",
}
if not os.path.exists("../plot/mortality/"):
    os.makedirs("../plot/mortality/")

for exper in explainers:
    if "gate_mask" not in exper:
        with open(path+exper+"_result_4_42.npy", 'rb') as file:
            array = np.load(file)
    else:
        with open("./results_gate/"+exper+"_result_0_42_plt.npy", 'rb') as file:
            array = np.load(file)
    for i in range(10):
        plot_example_box(array, i, "../plot/mortality/{}_{}.png".format(exper,i), k=25)


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


gs = GridSpec(2, 5)

# 创建子图并设置位置
axs = []
for i in range(2):
    for j in range(5):
        ax = plt.subplot(gs[i, j])
        ax.set_xticks([])
        ax.set_yticks([])
        axs.append(ax)

i=5
for kk, exper in enumerate(explainers):
    if "extr" in exper:
        i=4

    file_name = "../plot/mortality/{}_{}.png".format(exper,i)

    with Image.open(file_name) as im:
        axs[kk].imshow(im)
        axs[kk].set_title(names[exper],fontsize=6, va="center")

gs.update(wspace=0.2, hspace=-0.7)
# plt.show()
plt.savefig("../plot/pltmortality.pdf", dpi=800, bbox_inches='tight')



