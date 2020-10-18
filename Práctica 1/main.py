""" Imports """
from adaline import Adaline
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

""" Paths data """
current_folder = os.path.dirname(os.path.abspath(__file__))
files_initial_path = 'DatosProcesados/'
files_output_path = 'Salidas/'
file_extension = '.txt'



""" File to vector: no headers values to float """
def leer(directorio, fichero, tipo):
    # Abrimos el fichero
    with open(os.path.join(current_folder, directorio + fichero + tipo + file_extension), mode='r') as file_data:
        # Lo leemos
        reader = csv.reader(file_data, delimiter=',')
        #reader = np.array(reader)
        # Return vector of rows where each row is vector of float values
        # Nos saltamos las categorias
        #datos = []
        # Quitamos las cabeceras
        next(reader)
        return [[float(value) for value in row] for row in reader]
        """for i, fila in enumerate(reader):
            for j, valor in enumerate (fila):
                datos[i,j] = float(valor)
        return datos"""

""" Main """
def main():
    #problem = 'DatosEntrada'
    # Trainind and validation data
    datos_entrenamiento = leer(files_initial_path, 'DatosEntrada', '_entrenamiento')
    #datos_validacion = leer(files_initial_path, 'DatosEntrada', '_validacion')

    print(datos_entrenamiento)
    # Learning rates to test
    #learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]

    # Whole process start
    #test_learning_rates(problem, training, validation, learning_rates)


if __name__ == '__main__':
    main()


