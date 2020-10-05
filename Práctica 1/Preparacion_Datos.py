import csv
import random
from random import sample "Prueba Git Hub"

""" Directorios """
carpeta_DatosProcesados = 'DatosProcesados/'
DatosEntrada = 'DatosEntrada'
extension = '.dat'
extension_salida = ".txt"


""" Variables globales: porcentajes """
porcentaje_entrenamiento = 0.6
porcentaje_validacion = 0.2
porcentaje_test = 0.2



""" Función que lee los datos del fichero de entrada """
def lectura_DatosEntrada():
    datosEntrada= []
    #Primero abrimos el fichero
    with open(DatosEntrada + extension, mode='r') as FicheroEntrada:
         #Leemos cada campo, que esta separado por comas
        reader = csv.reader(FicheroEntrada, delimiter=',')
        #Guardamos en nuestra variable datosEntrada cada fila leída
        for fila in reader:
            datosEntrada.append(fila)
    return datosEntrada

""" Calculamos el valor mínimo de una columna """
def valor_minimo(datosEntrada,columna):
    #Definimos el valor minimo inicial como el primer valor encontrado en la columna
    minimo= float(datosEntrada[0][columna])
    for fila in datosEntrada:
        #Comparamos el valor de la fila que estamos recorriendo en la columna deseada y lo comparamos con el minimo actual
        if float(fila[columna]) < minimo:
            #Si ese valor es menor --> cambiamos este al valor minimo
            minimo = float(fila[columna])
    #Cuando recorramos y comparemos todos los valores de esa columna --> devolvemos el minimo encontrado
    return minimo

""" Calculamos el valor máximo de una columna """
def valor_maximo(datosEntrada,columna):
    #Definimos el valor máximo inicial como el primer valor encontrado en la columna
    maximo= float(datosEntrada[0][columna])
    for fila in datosEntrada:
        #Comparamos el valor de la fila que estamos recorriendo en la columna deseada y lo comparamos con el maximo actual
        if float(fila[columna]) > maximo:
            #Si ese valor es mayor --> cambiamos este al valor maximo
            maximo = float(fila[columna])
    #Cuando recorramos y comparemos todos los valores de esa columna --> devolvemos el maximo encontrado
    return maximo

""" Función para normalizar cada valor """
def normalizacion(dato, minimo, maximo):
    return float((dato - minimo)/(maximo - minimo))

""" Función para aleatorizar las filas """
def aleatorizacion(datos_sinCategoria):
    datos_aleatorios = sample(datos_sinCategoria,len(datos_sinCategoria))
    return datos_aleatorios


""" Función para guardar los valores máximos y mínimos de cada columna en un fichero (se utilizará posteriormente para desnormalizar) """
def save_max_min(categorias,datos_sinCategoria):
    lista_maximos= []
    lista_minimos= []
    #Para cada columna calculamos su minimo y maximo y lo guardamos y nuestra lista
    for columna in range(len(categorias)):
        lista_maximos.append(valor_maximo(datos_sinCategoria,columna))
        lista_minimos.append(valor_minimo(datos_sinCategoria,columna))
    #Generamos un fichero donde escribimos estos datos
    with open(carpeta_DatosProcesados + DatosEntrada + '_MaximoMinimo' + extension_salida, mode='w', newline='') as fichero_salida:
        writer = csv.writer(fichero_salida, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        #Escribimos los nombres de las columnas
        writer.writerow(categorias)
        #Debajo de cada columa escribimos los maximos
        writer.writerow(lista_maximos)
        #Debajo de cada columa escribimos los minimos
        writer.writerow(lista_minimos)
    
    return lista_maximos, lista_minimos


""" Funcion para generar los ficheros de salida de datos """
def generar_fichero(nombre, data, categorias, porcentaje, puntoInicial):
    #Abrimos el fichero donde vamos a escribir
    with open(carpeta_DatosProcesados + DatosEntrada + nombre + extension_salida, mode='w', newline='') as fichero_salida:
        writer = csv.writer(fichero_salida, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        #Escribimos las categorias
        writer.writerow(categorias)
        #Calculamos el número de filas a incluir en este fichero segun el porcentaje
        num_filas = int(porcentaje * len(data))
        puntoFinal = puntoInicial+num_filas
        print ("puntoIncial", puntoInicial, "puntoFinal", puntoFinal)
        #Escribimos en sus ficheros respectivos los datos
        for fila in range(puntoInicial, puntoFinal):
            writer.writerow(data[fila])
    return puntoFinal


""" Main """
def main():
    #Primero leemos los datos del fichero de entrada
    datos = lectura_DatosEntrada()
    #Generamos unos datos sin las categorías
    datos_sinCategoria = datos[1:]
    #Aleatorizamos los datos sin categorías
    datos_aleatorizados = aleatorizacion(datos_sinCategoria)

    #Guardamos los maximos y minimos de cada columna
    maximos, minimos = save_max_min (datos[0], datos_sinCategoria)
    
    #Calculamos los datos normalizados
    datos_normalizados = []
    for num_fila, contenidoCompleto_fila in enumerate(datos_aleatorizados):
        datos_normalizados.append([])
        for columna, valor in enumerate(contenidoCompleto_fila):
            datoNormalizado = normalizacion(float(valor), minimos[columna], maximos[columna])
            datos_normalizados[num_fila].append(datoNormalizado)

    print("datosConcategoria", len(datos))
    print("datosSincategoria", len(datos_normalizados))

    #Generamos los tres ficheros: Entrenamiento, validación y test 
    puntoFinal_entrenamiento = generar_fichero('_entrenamiento', datos_normalizados, datos[0], porcentaje_entrenamiento, 0)
    puntoFinal_validacion = generar_fichero('_validacion', datos_normalizados, datos[0], porcentaje_validacion, puntoFinal_entrenamiento)
    generar_fichero('_test', datos_normalizados, datos[0], porcentaje_test, puntoFinal_validacion)
   
if __name__ == '__main__':
    main()