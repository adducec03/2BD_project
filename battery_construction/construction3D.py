import json
import trimesh
import numpy as np
from trimesh.creation import cylinder
from trimesh.visual import color as color_utils


def convert_2d_to_3d(h,input,output):

    # CONFIGURAZIONE
    SEZIONI = 32    # Risoluzione cilindro
    ALTEZZA = h

    # COLORI per gruppo (RGBA)
    COLORI = [
        (102, 153, 204),  # azzurro tenue
        (204, 153, 102),  # sabbia chiara
        (153, 204, 153),  # verde salvia
        (204, 102, 102),  # rosso antico
        (153, 153, 204),  # lilla/grigio lavanda
        (102, 204, 204),  # verde acqua opaco
        (255, 204, 153),  # pesca chiaro
        (204, 153, 204),  # lilla più scuro
        (153, 102, 102),  # rosso mattone
        (102, 102, 153),  # blu lavanda scuro
        (153, 204, 255),  # celeste freddo
        (204, 255, 204),  # verde menta pallido
        (255, 255, 204),  # giallo burro
        (204, 204, 255),  # pervinca
        (255, 204, 204),  # rosa cipria
        (204, 255, 255),  # ghiaccio chiaro
        (204, 255, 153),  # lime spento
        (255, 229, 204),  # albicocca chiara
        (204, 204, 204),  # grigio medio
        (102, 153, 153)   # verde muschio polveroso 
    ]


    # Leggi il file JSON
    with open(input, 'r') as f:
        data = json.load(f)

    circles = data["circles"]
    gruppi = data["gruppi"]

    # Mappa cerchi → gruppo
    circle_to_group = {}
    for group_idx, group in enumerate(gruppi):
        for circle_idx in group:
            circle_to_group[circle_idx] = group_idx

    # Lista mesh
    all_meshes = []

    for i, circle in enumerate(circles):
        x, y = circle["center"]
        r = circle["radius"]

        gruppo = circle_to_group.get(i, 0)
        colore = COLORI[gruppo % len(COLORI)]

        # Crea cilindro centrato sull'origine
        cyl = cylinder(radius=r, height=ALTEZZA, sections=SEZIONI)

        # Applica colore al cilindro
        cyl.visual.vertex_colors = np.tile(colore, (len(cyl.vertices), 1))

        # Posiziona il cilindro
        cyl.apply_translation([x, y, ALTEZZA / 2])

        all_meshes.append(cyl)

    # Unisci tutti i cilindri
    scene = trimesh.util.concatenate(all_meshes)

    # Esporta GLB
    scene.export(output)
    print(f"✅ File GLB esportato: {output}")