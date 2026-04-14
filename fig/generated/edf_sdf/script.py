import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_table_csv(path):
    return np.loadtxt(path, delimiter=";")

def draw_pixel_values(image_path, table, output_path, decimals=1):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    h, w = img_np.shape[:2]
    if (h, w) != table.shape:
        raise ValueError(f"Image size {w}x{h} does not match table shape {table.shape[::-1]}")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.imshow(img_np, interpolation="nearest")

    # jemná mřížka mezi pixely
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    # schovat osy i hlavní tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    # odstranit okraj kolem obrázku
    for spine in ax.spines.values():
        spine.set_visible(False)

    # čísla do středu pixelů
    for y in range(h):
        for x in range(w):
            value = table[y, x]

            r, g, b = img_np[y, x] / 255.0
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "black" if brightness > 0.6 else "white"

            ax.text(
                x, y,
                f"{value:.{decimals}f}",
                ha="center",
                va="center",
                fontsize=15,
                color=text_color
            )

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

img_path = "binary_mask.png"
edt_table = load_table_csv("edf.csv")
sdf_table = load_table_csv("sdf.csv")

draw_pixel_values(img_path, edt_table, "edf_overlay.png", decimals=1)
draw_pixel_values(img_path, sdf_table, "sdf_overlay.png", decimals=1)