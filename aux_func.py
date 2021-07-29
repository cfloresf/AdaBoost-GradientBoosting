import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def plot_hist(df,var):
    ''' Crea gráficos de tipo distplot para la columna del dataframe que se desea, ploteando además la media y la mediana del dataset.

        Parameters:

        df (DataFrame): dataframe del cual se obtiene la data

        Returns:

        No return. Genera figura
    '''
    serie = df[var].dropna()
    media = np.mean(serie)
    mediana = np.median(serie)
    sns.distplot(serie, kde = True, color = 'green')
    plt.title('Histograma de la variable {}'.format(var))
    plt.axvline(x = media, linestyle = '-', color = 'tomato', label = 'Media {}'.format(var))
    plt.axvline(x = mediana, linestyle = '-', color = 'orange', label = 'Mediana {}'.format(var))
    plt.legend()
    plt.show()

def dummies(dataframe):
    ''' Genera variables dummies para las variables de tipo categóricas (type = object), eliminando aquella categoría
        recodificada que cuenta con la mayor cantidad de observaciones, generando una cantidad igual a n-1 variables
        dummy para cada variable categórica

        Parameres:
        dataframe (DataFrame): dataframe que se desea recodificar

        Returns:
        dataframe (DataFrame): dataframe con las nuevas columnas agregadas

    '''
    for colname, rowserie in dataframe.iteritems():
       if dataframe[colname].dtype=='O':
           if len(dataframe[colname].value_counts()) >= 2:
               cuenta = dataframe[colname].value_counts()
               largo = len(cuenta)
               n_dummies = largo -1
               for i in range(n_dummies,0,-1):
                   atributo_minoritario = cuenta.index[i]
                   nombre_variable = colname +'_'+ atributo_minoritario
                   dataframe[nombre_variable] = np.where(dataframe[colname] == atributo_minoritario,1,0)
               dataframe = dataframe.drop(colname, axis = 1)
    return dataframe

def variables(lista):
    ''' Concatena los valore dentro de una lista con el formato para generar modelos con statsmodels.formula.api

        Paramteres:
        lista (list): lista que contiene el nombre de las columnas de las variables o atributos del dataframe input para el modelo

        Returns:
        string (string): variables independientes concatenadas con '+'

    '''
    string = ''
    for index,value in enumerate(lista):
        if index < len(lista)-1:
            string += value + ' + '
        else:
            string += value
    return string

def regresion_depurada(df_dummies, lista, objetivo):
    ''' Genera un modelo de regresión con todos los atributos significativos para la variable dependiente a partir de un modelo saturado.

        Parameters:
        df_dummies (DataFrame): dataframe que contiene tanto las variables independientes y la variable dependiente a explicar.
        lista (list): lista que contiene los nombres de aquellas columnas que no forman parte de las variables independientes o atributos           del dataframe.
        objetivo (string): nombre de la variable dependiente a explicar por el modelo de regresión
    '''
    var_selected = df_dummies[df_dummies.columns.difference(lista)].columns.to_list()
    lista_string = variables(var_selected)
    modelo_ols = smf.ols('{} ~ {}'.format(objetivo,lista_string), data = df_dummies).fit()
    while any( x > 0.025 for x in modelo_ols.pvalues.to_list()):
        for i in range(len(modelo_ols.pvalues)):
            if modelo_ols.pvalues[i] >= 0.025:
                var = modelo_ols.pvalues.index[i]
                var_selected.remove(var)
        lista_string = variables(var_selected)
        modelo_ols = smf.ols('{} ~ {}'.format(objetivo,lista_string), data = df_dummies).fit()
    return modelo_ols

''' Genera un modelo de regresión logistico con todos los atributos significativos para la variable dependiente a partir de un modelo saturado.

    Parameters:
    df_dummies (DataFrame): dataframe que contiene tanto las variables independientes y la variable dependiente a explicar.
    lista (list): lista que contiene los nombres de aquellas columnas que no forman parte de las variables independientes o atributos           del dataframe.
    objetivo (string): nombre de la variable dependiente a explicar por el modelo de regresión logistico
'''
def modelo_depurado(df_dummies, lista, objetivo):
    var_selected = df_dummies[df_dummies.columns.difference(lista)].columns.to_list()
    lista_string = variables(var_selected)
    modelo_logit_depurado = smf.logit('{} ~ {}'.format(objetivo,lista_string), data = df_dummies).fit()
    while any( x > 0.025 for x in modelo_logit_depurado.pvalues.to_list()):
        for i in range(len(modelo_logit_depurado.pvalues)):
            if modelo_logit_depurado.pvalues[i] >= 0.025:
                var = modelo_logit_depurado.pvalues.index[i]
                var_selected.remove(var)
        lista_string = variables(var_selected)
        modelo_logit_depurado = smf.logit('{} ~ {}'.format(objetivo,lista_string), data = df_dummies).fit()
    return modelo_logit_depurado

def inverse_logit(x):
    return 1 / (1+np.exp(-x))
