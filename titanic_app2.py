#%% Bibliotecas Necessárias

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

#%% Configurando a página do app

st.set_page_config(page_title='Você sobreviveria ao desastre do Titanic?', 
                   page_icon='🚢', 
                   layout='wide'
                   )

#%% Configurando as abas do app

tab1, tab2 = st.tabs(['🔍 Simulador', 
                      'ℹ️ Sobre o app']
                     )
#%% Criando colunas para o tba1

with tab1:
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        #%% Apresentação do app
        st.title('🚢 Você sobreviveria ao desastre do Titanic?')
        st.subheader('Simule sua situação no Titanic e veja suas chances de sobrevivência segundo os dados reais de 1912!')
        st.subheader('Preencha o questionário e embarque nessa viagem para o passado!')
        
        #%% Inputs do usuário 
        
        # Sexo
        st.text('')
        sexo = st.selectbox('##### Qual seu sexo?',
                            ['...','Masculino', 'Feminino']
                            )
        
        # Idade
        st.text('')
        Age = st.number_input('##### Qual a sua idade?',
                                min_value=1, 
                                max_value= 100
                                )
        
        # Classe do navio
        st.text('')
        st.markdown('##### Com base na sua situação financeira atual, qual passagem você teria comprado?')
        classe = st.selectbox('**Escolha a Classe:**',
                              [
                                  '...',
         "1ª Classe – Luxo absoluto, dinheiro nunca é o problema.",
         "2ª Classe – Confortável, mas sem ostentar.",
         "3ª Classe – Simples, economizando ao máximo."
         ]
            )
        
        # Origem do navio
        st.text('')
        st.markdown('##### Imagine que o Titanic estivesse partindo do Brasil. Qual dessas cidades estaria mais próxima de onde você mora atualmente?')
        origem = st.selectbox('**Escolha a origem de partida:**',
                              [
                                  '...',
       "Southampton (Inglaterra) → Seria como embarcar por São Paulo.",
       "Cherbourg (França) → Seria como embarcar por Recife.",
       "Queenstown (Irlanda) → Seria como embarcar por Porto Alegre."
   ]
    )
        
        # Irmãos ou Cônjuge
        st.text('')
        st.markdown('##### Quantos familiares estariam com você (irmãos ou cônjuge)?')
        SibSp = st.number_input('**Nº de irmãos/cônjuge:**', 
                                min_value=0, 
                                max_value=10
                                )
        
        # Pais ou Filhos
        st.text('')
        st.markdown('##### Quantos pais ou filhos estariam viajando com você?')
        Parch = st.number_input('**Nº de pais/filhos:**', 
                                min_value=0, 
                                max_value=10
                                )
               
        #%% Transformação das variáveis para o modelo
        
        # Sexo
        if 'Masculino' in sexo:
            Sex = 1
        else:
            Sex = 0
            
        # Classe do navio
        if '1ª Classe – Luxo absoluto, dinheiro nunca é o problema.' in classe:
            Pclass = 1
        elif '2ª Classe – Confortável, mas sem ostentar.' in classe:
            Pclass = 2
        else:
            Pclass = 3
        
        # Origem do navio
        if 'Southampton (Inglaterra) → Seria como embarcar por São Paulo.' in origem:
            Embarked = 2
        elif 'Cherbourg (França) → Seria como embarcar por Recife.' in origem:
            Embarked = 0
        else:
            Embarked = 1
        
        # Valor da Passagem
        if Pclass == 1:
            Fare = 60.28
        elif Pclass == 2:
            Fare = 14.25
        else:
            Fare = 8.05
            
        #Tamanho da Família
        FamilySize = SibSp + Parch + 1
        
        # Estava sozinho
        IsAlone = int(FamilySize == 1)
        
        #%% Criando o Data Frame 
        
        input_df = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, IsAlone]],
                        columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone'])
        
        #%% Carregando o modelo
        
        with open('best_model_titanic.pkl', 'rb') as file:
            model = pickle.load(file)
            
        #%% Botão de ativação do simulador
        
        st.markdown('---')
        ativar = st.button('🔍 Verificar minhas chances de sobrevivência')
        
        if ativar:
            if sexo == '...':
                st.warning('Por favor, selecione seu sexo para continuar.')
            elif classe == '...':
                st.warning('Por favor, selecione sua classe para continuar.')
            elif origem == '...':
                st.warning('Por favor, selecione sua origem de partida para continuar.')
            else:
                # Predição
                prob = model.predict_proba(input_df)[0][1]

            if prob > 0.7:
                st.success(f"🎉 Alta chance de sobrevivência! ({prob:.2%})")
                st.image('https://media.tenor.com/2EOBvOkH_6AAAAAM/bird-calender.gif')
            elif prob >= 0.5:
                st.info(f"🤞 Chance moderada de sobrevivência: {prob:.2%}")
                st.image('https://media1.tenor.com/m/kWejy2kDcTwAAAAC/office.gif')
            else:
                st.error(f"💀 Baixa chance de sobrevivência: {prob:.2%}")
                st.image('https://media1.tenor.com/m/06KB-H2GctIAAAAd/the-office-michael-scott.gif')
                

#%% Entenda o resulado:

            with col2:
                st.title('📊 Entenda o Resultado:')
                
                # Gráfico das importâncias das variáveis 
                feature_importance = pd.DataFrame({'Variáveis':input_df.columns,
                                                   'Importância':model.feature_importances_
                                                   }).sort_values('Importância', ascending=False)
                
                st.markdown('##### Quais fatores mais influenciaram a sobrevivência no Titanic?')
                st.markdown('Este gráfico mostra as variáveis que o modelo considera mais relevantes para prever se uma pessoa sobreviveria ao naufrágio do Titanic. Quanto maior a barra, maior o peso da característica na decisão final do modelo. Por exemplo, o sexo e a classe do passageiro tiveram grande impacto na sobrevivência real – e o modelo capturou esse padrão.')
                fig_bar = px.bar(data_frame=feature_importance, x='Variáveis', y='Importância')
                st.plotly_chart(fig_bar)
                
                # Carregando o Dataset
                @st.cache_data
                def load_data():
                    return pd.read_csv('Titanic-Dataset-Final.csv')
                
                df = load_data()
                
                df_survived = df.groupby('Sex')['Survived'].value_counts().reset_index()
                df_survived['Sex'] = df_survived['Sex'].map({1:'Masculino', 0:'Feminino'})
                df_survived['Survived'] = df_survived['Survived'].map({1:'Sobreviveu', 0:'Não Sobreviveu'})
                df_survived.columns = ['Sexo', 'Status', 'Contagem']
                
                st.markdown('##### Qual foi a distribuição de sobreviventes por sexo?')
                fig_bar2 = px.bar(df_survived, x='Status', y='Contagem', color='Sexo')
                st.plotly_chart(fig_bar2)
            
#%% Sobre o app  

with tab2:
    
    st.subheader('🎯 Objetivo:')
    st.markdown('''######  Este aplicativo não é apenas uma ferramenta de Machine Learning - é uma máquina do tempo. Criado para te transportar ao cenário do trágico naufrágio do RMS Titanic, ele utiliza dados reais de 1912 para revelar qual teria sido seu destino naquela noite gelada no Atlântico Norte. Com base em características como sua idade, classe da passagem e origem da viagem, você poderá descobrir suas chances de sobrevivência e entender como diferentes fatores influenciaram o desfecho de milhares de passageiros.''')
    
    st.text('')
    st.subheader('🤖 Modelo de Machine Learning/Construção Web') 
    st.markdown('###### O modelo utilizado foi um **Random Forest Classifier**, uma técnica baseada em árvores de decisão que combina vários classificadores para melhorar a precisão da predição.')
    st.markdown('**Acurácia obtida nos dados de teste:** 84,35%')
    st.markdown('**ROC AUC Score obtida nos dados de teste:** 83,07%')
    st.markdown('###### Este app foi desenvolvido com **Streamlit**, uma poderosa ferramenta de código aberto para criação de interfaces web interativas em Python, ideal para projetos de Ciência de Dados e protótipos rápidos.')
    
    st.text('')
    st.subheader('🔍 Saiba mais') 
    st.markdown('###### O dataset utilizado foi o clássico conjunto de dados do Titanic, disponibilizado publicamente no 🔗[Kaggle](https://www.kaggle.com/competitions/titanic/overview).')
    st.markdown('###### Você pode acessar mais informações sobre os dados e até tentar criar seu próprio modelo.')
    
    st.text('')
    st.subheader('👨‍💻 Sobre o autor') 
    st.markdown('###### Olá! Meu nome é **Gabriel Silva Ferrando** e sou um entusiasta em início de jornada no mundo da Ciência de Dados e Machine Learning. Este projeto é uma das minhas primeiras explorações práticas, unindo aprendizado técnico com criatividade para tornar dados históricos mais acessíveis e interessantes.')
    st.markdown('###### Acompanhe meus próximos passos e projetos no GitHub: ')
    st.markdown('🔗[GitHub](https://github.com/GabrielFerrando)')
    st.markdown('🔗[LinkedIn](https://www.linkedin.com/in/gabriel-ferrando-ba543122a/)')
    
    
    
    
    
                            
                
    
        
        






        