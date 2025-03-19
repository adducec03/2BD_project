import math

#vincoli dimansioni (mm)
diameter=18         #diametro batteria
lenght=150          #larghezza rettangolo
height=220          #altezza rettangolo

#vincoli tecnici
tot_volts=11.1        #voltaggio batteria totale
tot_amps=5         #amperaggio batteria totale
cell_volts=3.7      #voltaggio singola cella
cell_amps=2.5         #amperaggio singola cella



#calcolo di quante batterie ci entrano
n_cell_x = lenght // diameter
n_cell_y = height // diameter
n_cell_tot = n_cell_x * n_cell_y

#calcolo di quante batterie in serie e quante in parallelo
n_cell_s=round(tot_volts/cell_volts)
n_cell_p=round(tot_amps/cell_amps)
print(f"il numero massimo di celle che possono entrare in questa forma è {n_cell_tot}")
print(f"per ottenere le specifiche richieste ti servono almeno {n_cell_s*n_cell_p} celle che devono essere collegate come {n_cell_s}S{n_cell_p}P")