import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import Delaunay, ConvexHull
import pandas as pd


def _circumradius(a, b, c):
    """Raggio circoscritto del triangolo ABC (coordinate 2D)."""
    ab = np.linalg.norm(b - a, axis=-1)
    bc = np.linalg.norm(c - b, axis=-1)
    ca = np.linalg.norm(a - c, axis=-1)
    s = (ab + bc + ca) / 2.0
    area = np.maximum(s * (s - ab) * (s - bc) * (s - ca), 0.0) ** 0.5  # formula di Erone
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (ab * bc * ca) / (4.0 * np.where(area > 0, area, np.nan))
    return R

def _alpha_shape_edges(points, alpha_radius):
    """
    Estrae gli spigoli di bordo dell'alpha-shape:
    tiene solo i triangoli con raggio circoscritto <= alpha_radius,
    e prende gli spigoli che appartengono a una sola di queste facce.
    """
    if len(points) < 3:
        return []
    tri = Delaunay(points)
    simplices = tri.simplices
    A = points[simplices[:, 0]]
    B = points[simplices[:, 1]]
    C = points[simplices[:, 2]]
    R = _circumradius(A, B, C)
    keep = np.isfinite(R) & (R <= alpha_radius)

    from collections import defaultdict
    edge_count = defaultdict(int)
    for s, k in zip(simplices, keep):
        if not k:
            continue
        edges = [(s[0], s[1]), (s[1], s[2]), (s[2], s[0])]
        for i, j in edges:
            e = tuple(sorted((int(i), int(j))))
            edge_count[e] += 1
    boundary = [e for e, cnt in edge_count.items() if cnt == 1]
    return boundary

def plot_cells_with_outline(coords, R=9.0,
                            method="convex", alpha_radius=None,
                            cell_face="white", cell_edge="black",
                            cell_lw=0.6, edge_lw=1.6, edge_color="black",
                            figsize=(12,8), invert_y=True,
                            title=None, show=True, save=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    coords = np.asarray(coords)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)

    # Celle
    for (x, y) in coords:
        ax.add_patch(Circle((x, y), R, facecolor=cell_face, edgecolor=cell_edge, linewidth=cell_lw, zorder=2))

    # (contorno opzionale; per semplicità, lo salto qui)

    xs, ys = coords[:,0], coords[:,1]
    ax.set_xlim(xs.min() - 2*R, xs.max() + 2*R)
    ax.set_ylim(ys.max() + 2*R, ys.min() - 2*R) if invert_y else ax.set_ylim(ys.min() - 2*R, ys.max() + 2*R)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=160)
        print(f"[salvato] {save}")
    if show:
        plt.show()
    return ax


df = pd.read_csv("new_version/out.csv")              # colonne: x,y
coords = df[['x','y']].to_numpy()

# 1) Contorno "alpha-shape" (aderente alla forma)
plot_cells_with_outline(
    coords, R=9.0,
    method="convex",
    title="Celle (solo cerchi)",
    show=True,                           # apri finestra interattiva
    save=None              # e salva anche su file
)

# 2) Contorno semplice (convex hull)
plot_cells_with_outline(coords, R=9.0, method="convex",
                        title="Celle + contorno (convex hull)")