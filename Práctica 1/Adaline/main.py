from adaline import Adaline
import csv
import os



""" Variables globales """
directorio_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_datosProcesados = 'DatosProcesados/'
carpeta_salidasProducidas = 'salidasProducidas/'
extension = '.txt'



""" Función para leer un fichero csv y devolverlo en una lista """
def leer(carpeta_datosProcesados, fichero):
    # Abrimos el fichero
    with open(os.path.join(directorio_actual, carpeta_datosProcesados + fichero + extension), mode='r') as fichero_csv:
        # Lo leemos
        csv_reader = csv.reader(fichero_csv, delimiter=',')
        # Quitamos las cabeceras
        next(csv_reader)
        return [[float(value) for value in row] for row in csv_reader]

""" Función que se encarga de escribir en ficheros los resultados del modelo entrenado """    
def escribir_modelo(adaline, fichero, razon_aprendizaje):
    # Creamos una carpeta por cada razón de aprendizaje
    directorio = carpeta_salidasProducidas + fichero + ' - ' + str(razon_aprendizaje) +'/'
    if not os.path.exists(os.path.join(directorio_actual, directorio)):
        os.mkdir(os.path.join(directorio_actual, directorio))
    # Llamamos a la función de escribir pesos en un fichero
    escribir_pesos(adaline, os.path.join(directorio_actual, directorio))
    # Realizamos el test a este modelo
    errores_test = test(adaline, os.path.join(directorio_actual, directorio))
    # Escribimos los errores de entrenamiento, validación y test en un fichero
    escribir_errores(adaline, os.path.join(directorio_actual, directorio), errores_test)

""" Función para escribir los pesos + umbral en un fichero """
def escribir_pesos (adaline, directorio):
    file = open(directorio + 'pesos.txt', "w")
    file.write('longitude   latitude   MedianAge  TRooms  TBedrooms   population   households  medIncome    umbral\n')
    for x in range(len(adaline.pesos)):
        file.write(str('{0:.5f}'.format(adaline.pesos[x]))+ '    ')
    file.write(str('{0:.5f}'.format(adaline.umbral))+ '    ')


""" Escribimos los errores del modelo """
def escribir_errores(adaline, directorio, errores_test):
    file = open(directorio + 'errores.txt', "w")
    file.write('Ciclo   TrainMSE   TrainMAE   ValidMSE   ValidMAE   TestMSE     TestMAE\n')
    file.write('----------------------------------------------------------------------- \n')
    for x in range(len(adaline.errores_entrenamiento)):
        file.write(str(x) + '       '+ str('{0:.5f}'.format(adaline.errores_entrenamiento[x][0]))+ '    '+str('{0:.5f}'.format(adaline.errores_entrenamiento[x][1]))+ '     '+str('{0:.5f}'.format(adaline.errores_validacion[x][0]))+ '    '+str('{0:.5f}'.format(adaline.errores_validacion[x][1])) )
        if (x != len(adaline.errores_entrenamiento)-1):
            file.write('\n')
        # Al final escribimos tambien los errores del test
        else:
            file.write('    '+str('{0:.5f}'.format(errores_test[0][0]))+ '    '+str('{0:.5f}'.format(errores_test[0][1])))

    print ("TrainMSE   TrainMAE   ValidMSE   ValidMAE   TestMSE     TestMAE")
    print(str('{0:.5f}'.format(adaline.errores_entrenamiento[adaline.ciclos][0]))+ '    '+str('{0:.5f}'.format(adaline.errores_entrenamiento[adaline.ciclos][1]))+ '     '+str('{0:.5f}'.format(adaline.errores_validacion[adaline.ciclos][0]))+ '    '+str('{0:.5f}'.format(adaline.errores_validacion[adaline.ciclos][1]))+ '    '+str('{0:.5f}'.format(errores_test[0][0]))+ '    '+str('{0:.5f}'.format(errores_test[0][1])))

""" Función que prueba el conjunto test en el modelo final """
def test (adaline, directorio):
    # Leemos el fichero de datos de test
    datos_test = leer(carpeta_datosProcesados, 'DatosEntrada_test')
    # Variables auxiliares
    valores_test_entrada = []
    valores_test_salidaEsperada = []
    salidasProducidas_test = []
    errores_test = []

     # x[i] --> valores de entrada.
    for fila in datos_test:
            valores_test_entrada.append(fila[:-1])
    # d[i] --> Valores esperados. Estamos guardando un único valor por fila
    for fila in datos_test:
        valores_test_salidaEsperada.append(fila[len(datos_test[0]) - 1])

    # Calculo la salida producida
    for x in range(len(valores_test_entrada)):
        salidasProducidas_test.append(adaline.salida_producida(valores_test_entrada[x]))
    # Guardamos los errores producidos en el test
    errores_test.append(adaline.error(valores_test_salidaEsperada, salidasProducidas_test))
    
    # Desnormalizo las salidas (producida y esperada)
    fichero_max_min = leer(carpeta_datosProcesados, 'DatosEntrada_maximominimo')
    max = fichero_max_min[0][len(fichero_max_min[0]) - 1]
    min = fichero_max_min[1][len(fichero_max_min[1]) - 1]
    salidaEsperada_desnormalizada = desnormalizar (valores_test_salidaEsperada, max, min)
    salidaProducida_desnormalizada = desnormalizar(salidasProducidas_test, max, min)

    # Escribo los resultados
    file = open(directorio + 'testSalidas.txt', "w")
    file.write('SalidaEsperada      SalidaProducida\n')
    for x in range(len(salidaEsperada_desnormalizada)):
        file.write(str('{0:.5f}'.format(salidaEsperada_desnormalizada[x]))+'        '+str('{0:.5f}'.format(salidaProducida_desnormalizada[x]))+'\n')

    # Devuelvo los errores producidos en el test para escribirlos en su fichero correspondiente
    return errores_test


# Función que desnormaliza una lista de valores
def desnormalizar (valores, max, min):
    # Lista de los valores desnormalizados
    salida = []
    for value in valores:
        salida.append (value * (max - min) + min)
    return salida

""" Función que prueba las distintas razones de aprendizaje en el modelo de Adaline """
def entrenamiento(nombre_problema, datos_entrenamiento, datos_validacion, razones_aprendizaje):
    # Para cada razon de entrenamiento realizamos un modelo
    for razon_aprendizaje in razones_aprendizaje:
        # Mostramos por consola que razon de entrenamiento estamos utilizando
        print(" -> Probando una razón de entrenamiento = " + str(razon_aprendizaje))
        # Llamamos a adaline para realizar el entrenamiento
        adaline = Adaline(id=nombre_problema, razon_aprendizaje=razon_aprendizaje, datos_entrenamiento=datos_entrenamiento, datos_validacion=datos_validacion).entrenamiento()
        # Escribimos el modelo obtenido de Adaline en ficheros salida
        escribir_modelo(adaline, nombre_problema, razon_aprendizaje)
        



""" Main """
def main():
    
    # Leemos los ficheros de entrenamiento y de validacion que hemos creado
    datos_entrenamiento = leer(carpeta_datosProcesados, 'DatosEntrada_entrenamiento')
    datos_validacion = leer(carpeta_datosProcesados, 'DatosEntrada_validacion')

    # Lista con las distintas razones de aprendizaje
    razones_aprendizaje = [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.05, 0.005, 0.0005]
    entrenamiento('DatosEntrada', datos_entrenamiento, datos_validacion, razones_aprendizaje)


if __name__ == '__main__':
    main()


