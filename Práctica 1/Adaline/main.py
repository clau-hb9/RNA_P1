from adaline import Adaline
import csv
import os
import numpy as np


""" Variables globales """
directorio_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_datosProcesados = 'DatosProcesados/'
carpeta_salidasProducidas = 'salidasProducidas/'
extension = '.txt'



""" Función para leer un fichero csv y devolvelo en un array """
def leer(carpeta_datosProcesados, fichero):
    # Abrimos el fichero
    with open(os.path.join(directorio_actual, carpeta_datosProcesados + fichero + extension), mode='r') as fichero_csv:
        # Lo leemos
        csv_reader = csv.reader(fichero_csv, delimiter=',')
        # Quitamos las cabeceras
        next(csv_reader)
        return [[float(value) for value in row] for row in csv_reader]
       


""" Test different learning rates to compare different models, sort them by Test MAE and find best of them"""
def test_learning_rates(nombre_problema, datos_entrenamiento, datos_validacion, razones_aprendizaje):
    #models = []
    # Para cada razon de entrenamiento realizamos un modelo
    for razon_aprendizaje in razones_aprendizaje:
        # Mostramos por consola que razon de entrenamiento estamos utilizando
        print("Probando una razón de entrenamiento = " + str(razon_aprendizaje))
        #print("  - Starts:",datetime.now().strftime("%H:%M:%S"))

        # Train model (change graph=True and/or debug=True if needed)
        adaline = Adaline(id=nombre_problema, razon_aprendizaje=razon_aprendizaje, datos_entrenamiento=datos_entrenamiento, datos_validacion=datos_validacion, graph=False, debug=False).entrenamiento()
        print(adaline.pesos)

""" Main """
def main():
    #problem = 'DatosEntrada'
    # Leemos los ficheros de entrenamiento y de validacion que hemos creado
    datos_entrenamiento = leer(carpeta_datosProcesados, 'DatosEntrada_entrenamiento')
    datos_validacion = leer(carpeta_datosProcesados, 'DatosEntrada_validacion')

    # Lista con las distintas razones de aprendizaje
    razones_aprendizaje = [0.0001]

    test_learning_rates('DatosEntrada', datos_entrenamiento, datos_validacion, razones_aprendizaje)


if __name__ == '__main__':
    main()


