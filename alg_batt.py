import math

#vincoli dimensioni (mm)
diameter=18         #diametro batteria
lenght=200          #larghezza rettangolo
height=50          #altezza rettangolo

#vincoli tecnici
tot_volts=11.1        #voltaggio batteria totale
tot_amps=5         #amperaggio batteria totale
cell_volts=3.7      #voltaggio singola cella
cell_amps=2.5         #amperaggio singola cella



#calcolo di quante celle entrano nel rettangolo
n_cell_x = lenght // diameter       #numero celle che entrano in lunghezza
n_cell_y = height // diameter       #numero celle che entrano in altezza
n_celle_max = n_cell_x * n_cell_y   #numero di celle massimo che entrano nel rettngolo

#calcolo di quante celle in serie e quante in parallelo
n_cell_s=round(tot_volts/cell_volts)    #celle in serie
n_cell_p=round(tot_amps/cell_amps)      #gruppi in parallelo
n_cell_tot=n_cell_p*n_cell_s            #numero di celle necessarie

print(f"il numero massimo di celle che possono entrare in questa forma è {n_celle_max}")
print(f"per ottenere le specifiche richieste ti servono almeno {n_cell_tot} celle")
if(n_cell_tot>n_celle_max):
    print("la richiesta non puo essere soddisfatta perche non hai abbastanza celle")
else:
    print(f"le celle devono essere collegate come {n_cell_s}S{n_cell_p}P")