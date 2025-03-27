import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LABEL_COLORS_NEW_EN = {
    "#2ca02c": "asphalt",
    "#1f77b4": "concrete",
    "#ff7f0e": "metal",
    "#d62728": "road marking",
    "#8c564b": "fabric, leather",
    "#7f7f7f": "glass",
    "#bcbd22": "plaster",
    "#ff9896": "plastic",
    "#17becf": "rubber",
    "#aec7e8": "sand",
    "#c49c94": "gravel",
    "#c5b0d5": "ceramic",
    "#f7b6d2": "cobblestone",
    "#c7c7c7": "brick",
    "#dbdb8d": "grass",
    "#9edae5": "wood",
    "#393b79": "leaf",
    "#6b6ecf": "water",
    "#9c9ede": "human body",
    "#637939": "sky"
}

# 设置图的尺寸
fig, ax = plt.subplots(figsize=(4, 6))

# 生成颜色图例
for i, (color, label) in enumerate(LABEL_COLORS_NEW_EN.items()):
    ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
    ax.text(1.2, i + 0.5, label, va='center', fontsize=10)

ax.set_xlim(0, 3)
ax.set_ylim(0, len(LABEL_COLORS_NEW_EN))
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

plt.show()