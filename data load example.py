import os
PATH = "C://Users\lok20\OneDrive\_Master\MAIA-ERASMUS//2 Semester\Interdiscipilanry Project AIA_ML_DL\GRAZPEDWRI-DX"

for dirpath, dirnames, filenames in os.walk(PATH):
    for fp in filenames:
        print(os.path.abspath(fp))