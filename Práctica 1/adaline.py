import matplotlib.pyplot as plt
import random
import numpy as np

#Implementación de la clase Adaline
class Adaline:

    #Declaración de los parámetros de la clase
    def __init__(self, id, razon_aprendizaje, datos_entrenamiento, datos_validacion, graph, debug, min_ciclos=3, max_ciclos=10):
        # id
        self.id = id 
        # Tiempo de entrenamiento
        self.max_ciclos = max_ciclos
        self.min_ciclos = min_ciclos
        # Razon de aprendizaje
        self.razon_aprendizaje = razon_aprendizaje
        # Datos
        self.datos_entrenamiento = datos_entrenamiento
        self.datos_validacion = datos_validacion
        # Definimos aleatoriamente los pesos y umbral incial
        self.pesos = []
        self.umbral = random.uniform(-1,1)
        for _ in range(len(self.datos_entrenamiento[0]) - 1):
            self.pesos.append(random.uniform(-1,1))

        # Do or do not show graph while training
        self.figure = plt.figure(id)
        self.graph = graph
        # Do or do not show debug logs while training
        self.debug = debug
        self.errors_log = ''


    def salida_producida(self, valores_entrenamiento_entrada):
        producto = []
        for num1, num2 in zip(self.pesos, valores_entrenamiento_entrada):
            producto.append(num1 * num2)
        suma = sum(x for x in producto)
        return suma

    def descenso_gradiante (self, salida_producida, valor_esperado, valores_entrenamiento_entrada):
        # Calculamos el incremento --> Razon de entrenamiento * (d[i]-y[i])
        delta = self.razon_aprendizaje * (valor_esperado - salida_producida)
        # Modificamos los pesos 
        for columna, peso in enumerate(self.pesos):
            self.pesos[columna] = peso + (delta * valores_entrenamiento_entrada[columna])
        # Modificamos el umbral
        self.umbral += delta


    # Returns whether there is a reason to stop training 
    def parada(self, errores_validacion):
        # There had not been significant changes (>0.0001) since [min_epochs] iterations before
        error_validacion_antiguo = errores_validacion[(len(errores_validacion) - 3), 1] 
        error_validacion_nuevo = errores_validacion[(len(errores_validacion -1 ), 1)]
        diferencia = error_validacion_antiguo - error_validacion_nuevo
        if (diferencia < 0.0001):
            no_cambios_significantes = True

        # Debug stop condition
        if self.debug and no_cambios_significantes:
            print("[STOP] No significant change | Trained epochs: ", self.ciclos)
        return no_cambios_significantes


    """ Computing MSE (mean square error) and MAE (mean absolute error) for current model state """
    def error(self, valores_esperados, salidasProducidas):
        diferencia = []
        #Para todos los resultados obtenidos calculamos su diferencia con los esperados
        for x in range(len(valores_esperados)):
            diferencia.append(valores_esperados[x] - salidasProducidas[x])
    
        suma_cuadratica = sum(x**2 for x in diferencia)
        mse = suma_cuadratica/len(diferencia)

        suma_absoluto = sum(abs(x) for x in diferencia)
        mae = suma_absoluto/len(diferencia)
        return mse, mae


    #Función de entrenamiento
    def entrenamiento(self):
        # Variables donde guardaremos los datos de entrada
        valores_entrenamiento_entrada = []
        valores_validacion_entrada = []
        valores_entrenamiento_salidaEsperada = []
        valores_validacion_salidaEsperada = []

        # x[i] --> valores de entrada. No leemos la columa final dado que es la salida esperada
        for fila in self.datos_entrenamiento:
            valores_entrenamiento_entrada.append(fila[:-1])
        for fila in self.datos_validacion:
            valores_validacion_entrada.append(fila[:-1])

        # d[i] --> Valores esperados. Estamos guardando un único valor por fila
        for fila in self.datos_entrenamiento:
            valores_entrenamiento_salidaEsperada.append(fila[len(self.datos_entrenamiento[0]) - 1])
        for fila in self.datos_validacion:
            valores_validacion_salidaEsperada.append(fila[len(self.datos_validacion[0]) - 1])
            
        # Error total producido en cada ciclo --> Contiene mse y mae para cada ciclo
        errores_validacion, errores_entrenamiento = [], []

        # Recorremos como máximo max_ciclos
        for i in range(self.max_ciclos):
            # Guardamos el numero de ciclos que vamos realizando hasta llegar a la condición de parada
            self.ciclos = i 
            # Una lista donde se almacenan las salidas producidas con los pesos finales de cada ciclo
            salidasProducidas_entrenamiento, salidasProducidas_validacion = [], []
                
            # Recorremos todas las entradas de un ciclo y ajustamos hasta finalizar el ciclo
            for j in range(len(self.datos_entrenamiento)):
                # Calculamos la salida producida para cada fila de entrenamiento
                salida_producida = self.salida_producida(valores_entrenamiento_entrada[j])
                # Con esta salida, procedemos a ajustar los pesos y el umbral
                self.descenso_gradiante(salida_producida, valores_entrenamiento_salidaEsperada[j], valores_entrenamiento_entrada[j])
            
            # Cuando tenemos ajustados los pesos de este ciclo, calculamos de nuevo la salida producida por cada entrada
            # Entrenamiento
            for x in range(len(valores_entrenamiento_entrada)):
                salidasProducidas_entrenamiento.append(self.salida_producida(valores_entrenamiento_entrada[x]))
            # Validacion
            for x in range(len(valores_validacion_entrada)):
                salidasProducidas_validacion.append(self.salida_producida(valores_validacion_entrada[x]))
                
            # Calculamos los errores mae y mse --> Esto solo se hara al final de cada ciclo
            errores_entrenamiento.append(self.error(valores_entrenamiento_salidaEsperada, salidasProducidas_entrenamiento))
            errores_validacion.append(self.error(valores_validacion_salidaEsperada, salidasProducidas_validacion))
                                
            # Break condition (after min_epochs at least)
            #if i > self.min_ciclos and self.parada(errores_validacion):
                #break

        # Return trained model
        return self











