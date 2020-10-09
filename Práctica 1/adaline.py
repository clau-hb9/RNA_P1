import matplotlib.pyplot as plt
import random
import numpy as np

#Implementación de la clase Adaline
class Adaline:


    #Declaración de los parámetros de la clase
    def __init__(self, id, razon_aprendizaje, datos_entrenamiento, datos_validacion, graph, debug, min_ciclos=500, max_ciclos=10000):
        # Identifier
        self.id = id #
        # Epochs
        self.max_ciclos = max_ciclos
        self.min_ciclos = min_ciclos
        # Eta
        self.razon_aprendizaje = razon_aprendizaje
        # Training data
        self.datos_entrenamiento = datos_entrenamiento
        # Validation data
        self.datos_validacion = datos_validacion
        # Do or do not show graph while training
        self.figure = plt.figure(id)
        self.graph = graph
        # Do or do not show debug logs while training
        self.debug = debug
        self.errors_log = ''
        # Initial weights and threshold
        self.pesos = random.uniform(-1,1) for x in range(len(self.datos_entrenamiento[0]) - 1)
        self.umbral = random.uniform(-1,1)



#Funcion entrenamiento
    def train(self):
        # Values (x[i])
        valores_entrenamiento_entrada = [fila[:-1] for fila in self.datos_entrenamiento]
        valores_validacion_entrada = [fila[:-1] for fila in self.datos_validacion]

        # salidas esperadas (d[i])
        valores_entrenamiento_esperados = [fila[len(self.datos_entrenamiento[0]) - 1] for fila in self.datos_entrenamiento]
        valores_validacion_esperados = [fila[len(self.datos_validacion[0]) - 1] for fila in self.datos_validacion]

        # errores
        errores_entrenamiento, errores_entrenamiento = [], []

        # ciclos entrenamiento
        for l in range(self.max_ciclos):

            # Current model epochs of training
            self.ciclos = l

            # For each training sample
            for j in range(len(self.datos_entrenamiento)):
                # Compute output (y[i])
                salida_producida = self.salida_producida(valores_entrenamiento_entrada[j])
                # Adjust weigths and threshold
                self.descenso_gradiante(valores_entrenamiento_esperados[j], salida_producida, valores_entrenamiento_entrada[j])

            # Training and validation errors (d[i] - y[i]) computed with last adjusted weights and threshold
            self.training_error = self.predict_set(training_values, training_expected)[1]
            self.validation_error = self.predict_set(validation_values, validation_expected)[1]

            # Training and validation errors
            training_errors.append(self.training_error)
            validation_errors.append(self.validation_error)

            log = self._errors_log()
            if self.debug:
                # Debugs in console new error variations
                print(log)

            self._graph(training_errors, validation_errors)
            if self.graph:
                # Draw graph with new error variations
                self.draw_graph()
            
            # Break condition (after min_epochs at least)
            if l > self.min_epochs and self.stop_condition(validation_errors):
                break

        # Return trained model
        return self



def descenso_gradiante (self, valores_entrenamiento_esperados, salida_producida, valores_entrenamiento_entrada):
	delta = self.razon_entrenamiento * (valores_entrenamiento_esperados - salida_producida)
	for columna, peso in enumerate(self.pesos):
		self.pesos[columna] = peso + delta * valores_entrenamiento_entradavalores_entrenamiento_entrada[columna]


def salida_producida(self, valores_entrenamiento_entrada):
	x1 = np.array(self.pesos)
	x2 = np.array(valores_entrenamiento_entrada)

	return x1.dot(x2)


