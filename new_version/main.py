import circle_packing as cp
import starting_k_grouping as kg
import series_ordering as so
from pathlib import Path
from shapely.geometry import Point, Polygon, LineString
import numpy as np
import networkx as nx


if __name__ == "__main__":

    json_file = "new_version/input.json"  # Path to the input JSON file
    out_csv = "new_version/out.csv"
    R = 9.0  # mm, radius of an 18650 seen from the top
    EPS = 0.2          # slack in adjacency threshold
    S = 30
    P = 10








    ###########################################################
    #Circle Packing
    #This module is used to pack the cells in a polygonal shape.
    ###########################################################

 # --------------------------------------------------- load outline ----
    poly, _ = cp.load_boundary(Path(json_file))

    # 1 ─ aligned hex grid (finer phase scan gives 1-3 extra cells)
    centres = cp.best_hex_seed_two_angles(poly, n_phase=16)
    print("hex grid :", len(centres))

    # pso refine (deleted for now, see below)
    #centres1 = pso_refine(centres0, poly)
    #print("PSO done")

    # 2 ─ first greedy insertion (fills obvious edge gaps)
    centres = cp.greedy_insert(poly, centres, trials=1000, max_pass=6)
    print("after greedy :", len(centres))

    # 3 ─ local compaction (Python re-implementation of Zhou’s batch-BFGS)
    centres = cp.batch_bfgs_compact(centres, R, poly, n_pass=4)
    print("after compaction :", len(centres))

    # 4 ─ second greedy pass (micron pockets now opened by compactor)
    centres = cp.greedy_insert(poly, centres, trials=1000, max_pass=3)
    print("final count :", len(centres))

    centres = cp.skeleton_insert(poly, centres, step=2.0)
    print("after skeleton :", len(centres))

    # ───▶  DIAGNOSTIC POCKET ANALYSIS  ◀─────────────────────────────────────
    free     = poly.buffer(-R).buffer(0)            # safe interior strip
    hull     = cp.unary_union([Point(x, y).buffer(R) for x, y in centres])
    residual = free.difference(hull)

    print(f"Area that could still host a centre: {residual.area:.1f} mm²")

    # Optional: largest empty circle (Shapely≥2.0 only)
    try:
        from shapely import maximum_inscribed_circle

        rad, centre_pt = cp.largest_empty_circle(residual)
        if rad > 0:
            print(f"Largest empty circle radius: {rad:.2f} mm "
                f"({2*rad:.2f} mm diameter)")
        else:
            print("Residual pocket too thin for even a tiny extra cell.")
    except ImportError:
        pass
    # ─────────────────────────────────────────────────────────────────────────

    # 5 ─ save + preview
    np.savetxt(out_csv, centres, delimiter=",", header="x,y", comments="")
    print(f"Saved {centres.shape[0]} circle centres to {out_csv}")
    #cp.plot_layout(poly, centres, title="Optimised layout")








    ###########################################################
    #K_grouping
    #This module is used to group the cells in k groups.
    ###########################################################

    centres_all = kg.load_centres(Path("new_version/out.csv"))
    N_all = len(centres_all)

    # ----- SELECT EXACTLY S*P CELLS ------------------------------------
    keep_mask = kg.drop_periphery_iterative(centres_all, S*P)
    centres   = centres_all[keep_mask]        # array of shape (S*P,2)
    print(f"Kept {len(centres)} cells, discarded {N_all-len(centres)} extras")

    # -------------------------------------------------------------------
    G   = kg.build_contact_graph(centres)        # only those kept cells
    #parts = metis_k_partition(G, S)           # S balanced parts
    #parts = geodesic_capacity_partition(G, S, P)
    #part_of = metis_connected_parts(G, S, P)   # healing on top of METIS

    # parts is a list (len = S*P).  Convert to dict {node_id: group_id}
    #part_dict = {i:part_of[i] for i in range(len(part_of))}
    kg.rb_exact_partition.gid_counter = 0
    part_of = kg.rb_exact_partition(G, list(G.nodes()), P)

    group_color = kg.color_groups_avoiding_adjacent(G, part_of, S)


    # ----- Sanity check -------------------------------------------------
    #group_sizes = [parts.count(g) for g in range(S)]
    #group_sizes = [list(part_of.values()).count(g) for g in range(S)]
    #print("Group sizes:", group_sizes)        # should all be P

    # ----- PLOT ---------------------------------------------------------
    #kg.plot_groups(poly, centres, part_of, S, group_color=group_color, title=f"S={S}, P={P} (adjacency-colored)")
    # ----------------------------------------------------------------------




    


    ###########################################################
    #Series Ordering
    #This module is used to order the groups in series.
    ###########################################################

    # After partitioning
    # part_of: dict node -> group  (if you have a list 'parts', convert: part_of = {i: parts[i] for i in range(len(parts))})

    #H = so.build_group_adjacency_graph(G, part_of, S)
    #print("Group graph connected components:", nx.number_connected_components(H))
    #order, segments = so.series_order_adjacent_walk(G, centres, part_of, S, prefer='near')


    order, group_edges = so.series_order_ortools(G, centres, part_of, beta=0.05, time_limit_s=5)
    print("Series order:", order)
    # Optional polarity (+1, -1 alternating)
    polarity = {g: (1 if i % 2 == 0 else -1) for i, g in enumerate(order)}
    print("Polarity:", polarity)

    # Plot
    group_color = kg.color_groups_avoiding_adjacent(G, part_of, S)
    so.plot_series_order(poly, centres, part_of, S, order, R=R)

