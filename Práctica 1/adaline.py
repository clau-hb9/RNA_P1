import matplotlib.pyplot as plt
# He ejecutado py -3 -m venv .venv
import random
import numpy as np

#Implementación de la clase Adaline
class Adaline:

    #Declaración de los parámetros de la clase
    def __init__(self, id, razon_aprendizaje, datos_entrenamiento, datos_validacion, graph, debug, min_ciclos=500, max_ciclos=10000):
        # Identifier
        self.id = id #
        # Tiempo de entrenamiento
        self.max_ciclos = max_ciclos
        self.min_ciclos = min_ciclos
        # Razon de aprendizaje
        self.razon_aprendizaje = razon_aprendizaje
        # Datos
        self.datos_entrenamiento = datos_entrenamiento
        self.datos_validacion = datos_validacion

        # Do or do not show graph while training
        self.figure = plt.figure(id)
        self.graph = graph
        # Do or do not show debug logs while training
        self.debug = debug
        self.errors_log = ''
        # Definimos aleatoriamente los pesos y umbral incial
        for x in range(len(self.datos_entrenamiento[0] - 1)):
            self.pesos[x] = random.uniform(-1,1) 
        self.umbral = random.uniform(-1,1)



#Función de entrenamiento
    def entrenamiento(self):
        # x[i] --> valores de entrada
        valores_entrenamiento_entrada = [fila[:-1] for fila in self.datos_entrenamiento]
        valores_validacion_entrada = [fila[:-1] for fila in self.datos_validacion]
        # d[i] --> Valores esperados
        valores_entrenamiento_esperados = [fila[len(self.datos_entrenamiento[0]) - 1] for fila in self.datos_entrenamiento]
        valores_validacion_esperados = [fila[len(self.datos_validacion[0]) - 1] for fila in self.datos_validacion]
        # Error total producido en cada ciclo --> Contiene mse y mae para cada ciclo
        # La ultima columna contendrá el mse y mae de todo el ciclo
        errores_validacion, errores_entrenamiento = [], []

        # Recorremos como máximo max_ciclos
        for i in range(self.max_ciclos):
            # Guardamos el ciclo en el que actualmente estamos
            self.ciclos = i
            # Una lista donde almacenaremos el error producido de cada entrada en este ciclo
            diferencia_entrenamiento, diferencia_validacion = [],[] 
            # Una lista donde se almacenan las salidas producidas en los pesos finales del ciclo
            salidas_entrenamiento, salidas_validacion = [], []
            
            # Recorremos todas las entradas de un ciclo y ajustamos hasta finalizar el ciclo
            for j in range(len(self.datos_entrenamiento)):
                # Calculamos la salida producida para cada dato de entrenamiento
                salida_producida = self.salida_producida(valores_entrenamiento_entrada[j])
                # Con esta salida, procedemos a ajustar los pesos y el umbral
                self.descenso_gradiante(salida_producida, valores_entrenamiento_esperados[j], valores_entrenamiento_entrada[j])
           
            # Cuando tenemos ajustados los pesos de este ciclo, calculamos de nuevo la salida producida por cada entrada
            # Entrenamiento
            for x in len(valores_entrenamiento_entrada):
                salidas_entrenamiento[x] = self.salida_producida(valores_entrenamiento_entrada[x])
            # Validacion
            for x in len(valores_validacion_entrada):
                salidas_validacion[x] = self.salida_producida(valores_validacion_entrada[x])
            # Entrenamiento
            for x in len(salidas_entrenamiento):
                diferencia_entrenamiento [x] = (valores_entrenamiento_esperados[x] - salidas_entrenamiento[x] )
            # Validacion
            for x in len(salidas_validacion):
                diferencia_validacion [x] = (valores_validacion_esperados[x] - salidas_validacion[x] )

            # Calculamos los errores mae y mse --> Esto solo se hara al final de cada ciclo
            errores_entrenamiento [i,j] = self.error(diferencia_entrenamiento)
            errores_validacion [i,j] = self.error(diferencia_entrenamiento)
            

            """log = self._errors_log()
            if self.debug:
                # Debugs in console new error variations
                print(log)

            self._graph(training_errors, validation_errors)
            if self.graph:
                # Draw graph with new error variations
                self.draw_graph()"""
            
            # Break condition (after min_epochs at least)
            """if i > self.min_ciclos and self.stop_condition(errores_validacion):
                break"""

        # Return trained model
        return self



def descenso_gradiante (self, salida_producida, valores_esperado, valores_entrenamiento_entrada):
	# Calculamos el incremento --> Razon de entrenamiento * (d[i]-y[i])
    delta = self.razon_entrenamiento * (valores_esperado - salida_producida)
	# Modificamos los pesos 
    for columna, peso in enumerate(self.pesos):
	    self.pesos[columna] = peso + (delta * valores_entrenamiento_entrada[columna])
    # Modificamos el umbral
    self.umbral += delta

def salida_producida(self, valores_entrenamiento_entrada):
	x1 = np.array(self.pesos)
	x2 = np.array(valores_entrenamiento_entrada)
	return x1.dot(x2)


""" Computing MSE (mean square error) and MAE (mean absolute error) for current model state """
def error(diferencia):
    suma_cuadratica = sum(x**2 for x in diferencia)
    mse = suma_cuadratica/len(diferencia)

    suma_absoluto = sum(abs(x) for x in diferencia)
    mae = suma_absoluto/len(diferencia)
    return mse, mae

""" Returns whether there is a reason to stop training 
    def stop_condition(self, validation_errors):
        # There had not been significant changes (>0.0001) since [min_epochs] iterations before
        no_significant_change = validation_errors[len(validation_errors) - (self.min_epochs - 1)][1] - validation_errors[-1][1] < 0.0001
        # Debug stop condition
        if self.debug and no_significant_change:
            print("[STOP] No significant change | Trained epochs: ", self.epochs)
        # Return true in case one of the two conditions is true
        return no_significant_change"""

