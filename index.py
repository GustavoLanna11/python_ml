#Importação da biblioteca pandas (que tras consigo funções de ML)
import pandas as pd 

#Importação do arquivo que será analisado
arquivo = pd.read_csv('C:/Users/gustavo.marques/Downloads/wine_dataset.csv')

#transformando variáveis string em int (no caso o tipo do vinho)
arquivo.head()
arquivo['style'] = arquivo['style'].replace('red',0)
arquivo['style'] = arquivo['style'].replace('white',1)

#separando as variáveis entre preditoras e alvo
#x = tipo do vinho; y = outros dados do vinho;
y, x = arquivo['style'], arquivo.drop('style', axis = 1 )

#início do Machine Learning, importando funções
from sklearn.model_selection import train_test_split

#criação dos conjuntos de treino e teste (30% dos dados serão o teste, enquanto 70% serão treino pra IA)
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size = 0.3)

#Criação do modelo para testar e treinar:
from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(x_treino,y_treino)

#Resultados: Apresentação dos resultados de teste + % de precisão da IA
resultado = modelo.score(x_teste, y_teste)
print("Porcentagem de Precisão: ", resultado)

#teste com dois perídos de dados
y_teste[400:410], x_teste[400:410]
previsoes = modelo.predict (x_teste[400:410])
print(previsoes)
