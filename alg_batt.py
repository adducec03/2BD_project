import math

#vincoli dimansioni (mm)
diameter=18         #diametro batteria
lenght=150          #larghezza retttangolo
height=220          #altezza rettangolo

#vincoli tecnici
tot_volts=36        #voltaggio batteria totale
tot_amps=20         #amperaggio batteria totale
cell_volts=4.2      #voltaggio singola cella
cell_amps=3.5       #amperagggio singola cella



#calcolo di quante batterie ci entrano
n_cell_x = lenght // diameter
n_cell_y = height // diameter
n_cell_tot = n_cell_x * n_cell_y

#calcolo di quante batterie in serie e quante in parallelo
n_cell_s=tot_volts//cell_volts
n_cell_p=tot_amps//cell_amps

print(f"hai bisogno di {n_cell_tot} celle che devono essere disposte in {n_cell_s}S{n_cell_p}P")
