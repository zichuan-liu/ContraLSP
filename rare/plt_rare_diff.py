import pickle
from utils.tools import print_results, plot_example_box
import os
datasets = "rare_time_diffgroup"
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
    for i in range(48,52):
        plot_example_box(array, i, "../plot/rare2plot/"+datasets+"/{}_{}.png".format(exper,i), k=10)


from PIL import Image
import numpy as np
from utils.tools import print_results, plot_example_box
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.axis("off")

# 创建一个4x2的GridSpec
gs = GridSpec(4, 8)
for kk, exper in enumerate(explainers):
    for i in range(48,52):
        ax = plt.subplot(gs[i-48, kk])
        with Image.open("../plot/rare2plot/"+datasets+"/{}_{}.png".format(exper,i)) as im:
            ax.imshow(im)
        if i==48:
            ax.set_title(names[exper], fontsize=6)
        if kk==0:
            if i==48 or i==49:
                ax.text(-0.3, 0.5, r"Group $S_1$", transform=ax.transAxes, fontsize=6, va='center', ha='center')
            else:
                ax.text(-0.3, 0.5, r"Group $S_2$", transform=ax.transAxes, fontsize=6, va='center', ha='center')
        ax.set_xticks([])
        ax.set_yticks([])


# 调整子图之间的间距
# plt.tight_layout()
gs.update(wspace=0.1, hspace=-0.65)
# plt.tight_layout()
# 显示图形
# plt.show()
plt.savefig(f"../plot/plt{datasets}.pdf", dpi=800, bbox_inches='tight')



