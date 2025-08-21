import circle_packing as cp
import starting_k_grouping as kg
import series_ordering as so
import parallel_plate as pp
import series_plate as sp
import thermo as st


from pathlib import Path
from shapely.geometry import Point, Polygon, LineString
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt




if __name__ == "__main__":


    ###########################################################
                            #Variables
    ###########################################################

    json_file = "new_version/input_std.json"  # Path to the input JSON file
    out_csv = "new_version/out.csv"
    R = 9.0  # mm, radius of an 18650 seen from the top
    EPS = 0.2          # slack in adjacency threshold
    S = 10
    P = 40








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

    prev = len(centres)
    t0 = cp.perf_counter()
    centres = cp.skeleton_insert(poly, centres, step=2.0)
    t1 = cp.perf_counter()
    cp.plot_phase(poly, centres, added_idx=np.arange(prev, len(centres)),
            title="After skeleton insert", r=R, time_s=t1 - t0)

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
    #roup_sizes = [parts.count(g) for g in range(S)]
    #group_sizes = [list(part_of.values()).count(g) for g in range(S)]
    #print("Group sizes:", group_sizes)        # should all be P

    # ----- PLOT ---------------------------------------------------------
    kg.plot_groups(poly, centres, part_of, S, group_color=group_color, title=f"S={S}, P={P} (adjacency-colored)")
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











    ##########################################################################
                                #Plates design
    #This module is used to design plates for parallel and series connections.
    ##########################################################################

    #-----------------------------------Parallel------------------------------
    plates = pp.make_all_plates(G, centres, part_of, S,
                                R=R, land=1.2, gap=2.0, outline=poly, res=64)
    plates = {g: pp.smooth_plate(poly, r_open=3, r_close=1, gap=2.0, safety=0.10)
          for g, poly in plates.items()}
    #plate = npb.smooth_plate(plates, r_open=0.8, r_close=0.4, gap=2.0, safety=0.10)
    welds  = pp.weld_points_all(G, centres, part_of, S, R=R, offset=2.6)
    # Plot
    pp.plot_plates(poly, centres, part_of, S, plates, welds, R=R,
                title=f"Nickel plates per group (S={S}, P={P}) – disjoint")


    #-----------------------------------Series--------------------------------
    # known:
    #   plates : dict {group -> shapely Polygon}  (parallel plates, disjoint)
    #   part_of, centres, S
    #   order  : list of group-ids from OR-Tools path
    #   I_pack : (estimated) peak pack current

    group_edges = [(order[i], order[i+1]) for i in range(len(order)-1)]


    bridges = sp.make_series_band_from_cells(
        G, centres, part_of, group_edges,
        R=9.0,
        rows=1,        # try 2–3 to match your photo (more rows ⇒ wider plate)
        inset=0.6,
        smooth=1.0,
        clip_poly=poly
    )

    sp.plot_series_plates_over_cells(poly, centres, part_of, S, bridges, R=9.0)


    #######################################################################################
                                        #Thermal analysis
    #This module is used to make a thermal analysis based on the specific battery topology
    #######################################################################################


    # ... after you have: G, part_of, order, S, P ...
    results = st.simulate_series_electrothermal(
        G, part_of, order, P,
        I_pack=80.0,            # choose the pack current you care about
        r_cell=0.020,           # 20 mΩ per cell (edit for your cell type)
        r_contact=0.0005,       # 0.5 mΩ per touching cell-cell pair
        r_extra=0.0000,         # extra nickel path (if you want)
        Rth_group=2.0,          # K/W to ambient for each group
        Rth_iface=0.7,          # K/W to ambient for each interface region
        Iperpair_max=10.0       # allowed current per contact pair
    )

    print(f"Total pack resistance: {results['R_pack']*1e3:.2f} mΩ")
    print(f"Pack drop @I:         {results['V_drop']:.2f} V")
    print(f"Pack electrical loss: {results['P_loss']:.2f} W")

    print("\nInterfaces (a->b):  w   R[mΩ]  P[W]  ΔT[K]  overload?")
    for (a, b), w, R, Pif, dT, ov in zip(
        results['interfaces']['edges'],
        results['interfaces']['w'],
        results['interfaces']['R'],
        results['interfaces']['P'],
        results['interfaces']['dT'],
        results['interfaces']['overload']
    ):
        print(f"{a:2d}->{b:2d}:  {w:3d}  {R*1e3:6.3f}  {Pif:6.2f}  {dT:6.2f}  {ov}")

    print("\nGroups g: R[mΩ], P[W], ΔT[K]")
    for g, (Rg, Pg, dTg) in enumerate(zip(
        results['groups']['R'],
        results['groups']['P'],
        results['groups']['dT']
    )):
        print(f"{g:2d}: {Rg*1e3:6.3f}  {Pg:6.2f}  {dTg:6.2f}")

    print("group_contact_counts\n")
    # Compute the contact counts once
    counts = st.group_contact_counts(G, part_of)

    print("\nprint_all_group_contacts\n")
    # Print all touching counts (optional)
    st.print_all_group_contacts(counts, S)

    print("\nprint_series_contacts\n")
    # Print only the counts for the series path
    st.print_series_contacts(order, counts)


    # 1) compute thermal data
    edges, groups, totals = st.compute_series_thermal(
        G, centres, part_of, order, S, P,
        I_pack=80.0,             # set as you wish
        r_cell_mohm=20.0,
        r_pair_mohm=0.5,
        Rtheta_group_KW=2.0,
        Rtheta_iface_KW=0.7,
        Iperpair_max=10.0
    )

    print(f"R_total = {totals['R_total_mohm']:.2f} mΩ, "
        f"V_drop = {totals['V_drop_V']:.2f} V, "
        f"P_loss = {totals['P_loss_W']:.1f} W")

    # 2) plot
    st.plot_series_thermal(poly, centres, part_of, S, order, edges, groups, R=R)




    


