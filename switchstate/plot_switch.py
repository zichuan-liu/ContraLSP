from PIL import Image
import numpy as np
from utils.tools import print_results, plot_example_box
import os
path = './results/'
explainers = [
            "occlusion",
            "augmented_occlusion",
            "integrated_gradients",
            "gradient_shap",
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

if not os.path.exists("../plot/switch/"):
    os.makedirs("../plot/switch/")

with open(path+"true.npy", 'rb') as file:
    lab_array = np.load(file)
    for i in range(30, 50):
        plot_example_box(lab_array, i, "../plot/switch/{}_{}.png".format("true", i), k=25)

for exper in explainers:
    with open(path+exper+"_result_4_42.npy", 'rb') as file:
        array = np.load(file)
        if exper == "gate_mask":
            array[:, :30] = lab_array[:, :30]
    for i in range(30, 50):
        plot_example_box(array, i, "../plot/switch/{}_{}.png".format(exper,i), k=25)


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 创建一个2x2的子图网格
gs = GridSpec(len(explainers)+1, 1)

# 创建子图并设置位置
axs = []
for i in range(len(explainers)+1):
    ax = plt.subplot(gs[i, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    axs.append(ax)

i=18    # 附录18， 正文的33
for kk, exper in enumerate(explainers):
    file_name = "../plot/switch/{}_{}.png".format(exper,i)
    with Image.open(file_name) as im:
        axs[kk].imshow(im)
        axs[kk].text(-5, 25, names[exper], ha='right', fontsize=10, va="center")

file_name = "../plot/switch/{}_{}.png".format("true", i)
with Image.open(file_name) as im:
    axs[-1].imshow(im)
    axs[-1].text(-5, 25, "Label", ha='right', fontsize=10, va="center")

# 调整子图之间的水平和垂直间隔
gs.update(wspace=0.2, hspace=0.1)

# 显示图形
# plt.show()
plt.savefig("../plot/pltswitch_f.pdf", dpi=800, bbox_inches='tight')



