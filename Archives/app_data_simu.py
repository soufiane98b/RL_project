import streamlit as st
import pandas as pd
import pickle
import os
import linucb
import matplotlib.pyplot as plt

################################################################################################
def reshape_dictionnaire_data(data):
    del data['Other Information']['distance entre paires groupes ']
    #if 'Policy' in data['Inforamtion sur la Simulation'].keys():
        #del data['Inforamtion sur la Simulation']['Policy']
    #print('ok')
    return data

def ordonner(Datas):
    return sorted(Datas, key=lambda x: int(x[1][-2:]))

def display_dict(dict_obj,name):
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            st.subheader(f" {key} ")
            display_dict(value,name)
        elif isinstance(value, pd.DataFrame):
            if key == 'df_encoded':
                st.write(f"* {key} : ")
                st.dataframe(value.rename(columns={'cluster': 'id_groupe'}))
            elif key == 'Information sur Reward et execution':
                st.write(f"* {key} : ")
                st.dataframe(value)
                # FOR CTR
                n = name.split('_')
                tmp = '\ncolumn = '+n[2]+'| max nb of categories = '+n[4]+'| nb of groups = '+n[6]
                st.write("* Graphiques de comparaison du CTR :")
                ax = value.drop('Execution time').plot.bar(figsize=(5, 3), rot=0, fontsize=3)
                ax.legend(fontsize=4)
                ax.set_title("Total Cumulative reward before and after shift"+tmp)
                st.pyplot(plt.gcf())
                # FOR TIME EXECUTION
                st.write("* Comparaison du temps d'execution :")
                ax = value.loc[['Execution time']].plot.bar(figsize=(5, 3), rot=0, fontsize=3)
                ax.legend(fontsize=4)
                ax.set_title("Time Execution"+tmp)
                st.pyplot(plt.gcf())

            else :
                st.write(f"* {key} : ")
                st.dataframe(value)

        else:
            if key == 'Aligned CTR':
                n = name.split('_')
                tmp = '\ncolumn = '+n[2]+'| max nb of categories = '+n[4]+'| nb of groups = '+n[6]
                st.write("* Graphiques de comparaison du CTR :")
                fig, ax = plt.subplots()
                for v in value : 
                    ax.plot(v[0],label=v[1])
                ax.set_title("Evolution of CTR before and after shift"+tmp)
                ax.legend(fontsize=4)
                fig.set_figwidth(5)
                fig.set_figheight(3)
                st.pyplot(fig)

                
            else :
                st.write(f"* {key} : {value}")


def app(Data):
    display_dict(Data[0],Data[1])

################################################################################################
# Recuperation de la donnée à configurer selon l'environnement 
#chemin = '/Users/soufiane/Documents/GitHub/universe/My_universe/Data_With_Simu_RL_fast_proba_cluster'
chemin = '/Users/soufiane/Documents/GitHub/universe/My_universe/TT'
fichiers = os.listdir(chemin)

#fichiers = ['DataSimuRl_c_10_f_12_g_10','DataSimuRl_c_5_f_5_g_10','DataSimuRl_c_5_f_12_g_10','DataSimuRl_c_10_f_5_g_10'] ###### A ENLEVER 
#fichiers =['DataSimuRl_c_15_f_12_g_30']
# On stock la donnée dans liste avec nom du fichier
Datas_simu = []
for nom_fichier in fichiers:
    with open(chemin+'/'+nom_fichier, 'rb') as f:
        #print(nom_fichier)
        data = pickle.load(f)
        data = reshape_dictionnaire_data(data)
        Datas_simu.append((data,nom_fichier))

Datas_simu = ordonner(Datas_simu)

#print('FINNNN')

"""
# Configuration de la page
st.set_page_config(page_title='info_Datas_'+str(len(Datas_simu)), layout="wide",page_icon=":chart_with_upwards_trend:")


options = ['Option 1', 'Option 2', 'Option 3']
choix = st.selectbox("Sélectionnez une option", options)
nom_utilisateur = st.text_input("Entrez votre nom")
if st.button("Cliquez ici"):
    st.write("Le bouton a été cliqué !")"""

# Configuration du Sommaire
st.sidebar.title("Sommaire")

i = 1
for d in Datas_simu:
    tmp = "Données du Data "+d[1]
    st.sidebar.markdown("- ["+tmp+"](#section-"+str(i)+")")
    st.markdown("<a name='section-"+str(i)+"'></a>", unsafe_allow_html=True)#
    st.markdown(f"## {tmp}")
    app(d)
    i=i+1



st.sidebar.markdown("---")
st.sidebar.markdown("Cliquez sur les liens ci-dessus pour accéder aux sections correspondantes.")

