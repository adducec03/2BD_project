import os

# Percorso della cartella con le foto
cartella = "photos"  # <-- CAMBIA QUI

# Ottieni solo i file .jpg nella cartella
file_foto = [f for f in os.listdir(cartella) if f.lower().endswith('.jpg')]
file_foto.sort()  # Ordina per nome (puoi usare .sort(key=os.path.getctime) per ordine cronologico)

# Rinomina i file con numerazione progressiva
for i, nome_file in enumerate(file_foto, start=1):
    estensione = os.path.splitext(nome_file)[1]
    nuovo_nome = f"{i}{estensione}"
    percorso_vecc = os.path.join(cartella, nome_file)
    percorso_nuovo = os.path.join(cartella, nuovo_nome)
    os.rename(percorso_vecc, percorso_nuovo)

print("Rinominati tutti i file con successo.")