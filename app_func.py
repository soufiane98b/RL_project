import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def ordonner(Datas):
    return sorted(Datas, key=lambda x: int(x[1][-2:]))

def display_dict(dict_obj,name):
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            st.subheader(f" {key} ")
            display_dict(value,name)
        elif isinstance(value, pd.DataFrame):
            if key == 'Information sur Reward et execution':
                st.write(f"* {key} : ")
                st.dataframe(value)
                # FOR CTR
                n = name.split('_')
                tmp = '\ncolumn = '+n[1]+'| max nb of categories = '+n[2]+'| nb of groups = '+n[3]
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
                tmp = '\ncolumn = '+n[1]+'| max nb of categories = '+n[2]+'| nb of groups = '+n[3]
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


def formulaire():
    col = st.multiselect("Sélectionnez les colonnes", list(range(1, 30)), [6])
    cat = st.multiselect("Sélectionnez le nombre de catégorie maximum pour chaque colonne", list(range(1, 30)), [9])
    gps = st.multiselect("Sélectionnez le nombre de groupes", list(range(5,71,5)), [10])
    nb_lignes = st.number_input("Saisissez le nombre de lignes", min_value=100, step=1)
    ratio = st.slider("Sélectionnez le ratio d'équilibre entre les groupes", 0.0, 1.0, step=0.01)
    dist = st.selectbox("Sélectionnez votre distribution de déséquilibre entre les groupes", ['uniform', 'poisson', 'gamma', 'pareto', 'lognorm', 'expon'],index=0)
    options = list(range(2, 50))
    nb_cluster = st.selectbox("Sélectionnez le nombre de cluster", options, index=0)
    nb_arm = st.selectbox("Sélectionnez le nombre d'arm", list(range(2,50,1)),index=8)
    
    return col, cat, gps, nb_lignes, ratio, dist,nb_cluster,nb_arm




