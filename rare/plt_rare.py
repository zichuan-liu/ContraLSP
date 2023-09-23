import pickle
from utils.tools import print_results, plot_example_box
import os
datasets = "rare_time"
path = './results/'+datasets+"/"
explainers = ["fo", "fp", "ig", "shap",
              "dynamask", "nnmask", "gatemask", "true"]
names = {
    "fo":"FO",
    "fp":"AFO",
    "ig":"IG",
    "shap":"SVS",
    "dynamask":"Dynamask",
    "nnmask":"Extrmask",
    "gatemask":"ContraLSP",
    "true":"Label",
}

if not os.path.exists("../plot/rare2plot/"+datasets+"/"):
    os.makedirs("../plot/rare2plot/"+datasets+"/")

for exper in explainers:
    with open(path+exper+"_saliency_1.pkl", 'rb') as file:
        array = pickle.load(file)
    for i in range(49,52):
        plot_example_box(array, i, "../plot/rare2plot/"+datasets+"/{}_{}.png".format(exper,i), k=10)


from PIL import Image
import numpy as np
from utils.tools import print_results, plot_example_box
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 创建一个4x2的GridSpec
gs = GridSpec(2, 4)
i = 50
for kk, exper in enumerate(explainers):
    ax = plt.subplot(gs[kk // 4, kk % 4])
    with Image.open("../plot/rare2plot/"+datasets+"/{}_{}.png".format(exper,i)) as im:
        ax.imshow(im)
    ax.set_title(names[exper], fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])

# 调整子图之间的间距
# plt.tight_layout()
gs.update(wspace=0.2, hspace=-0.4)
# plt.tight_layout()
# 显示图形
# plt.show()
plt.axis("off")

plt.savefig(f"../plot/plt{datasets}.pdf", dpi=800, bbox_inches='tight')



