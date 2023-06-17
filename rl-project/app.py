import streamlit as st
import dataGenerator
import simulator
import app_func

############################# Creation Formulaire ####################################################

# Configuration de la page
st.title("Paramètres de la simulation")
col, cat, gps, nb_lignes, ratio, dist,nb_cluster,nb_arm = app_func.formulaire()
button_clicked = st.button("Valider les paramètres de simulation")

############################# Creation Data et Simulation à partir du Formulaire ########################
if button_clicked:
    #------  Generation Data   ------
    generator = dataGenerator.generator_multiple_df(col,cat,gps,nb_lignes,ratio,dist,nb_cluster)
    st.write("En cours de generation de vos données")
    progress_bar = st.progress(0)
    status_text = st.empty()
    Result = []
    while True:
        my_next_value = next(generator)
        if my_next_value==-1:
            break
        # Affichage Avancement sur Console
        status_text.text(f"{my_next_value[0]} sur {my_next_value[1]}")
        progress_bar.progress(my_next_value[0]/my_next_value[1])
        Result.append((my_next_value[2],my_next_value[3]))

    #------  Lancement de la Simulation   ------

    st.write("Fin de generation de vos données et lancement de la simulation")
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(len(Result)):
        r = Result[i]
        info = simulator.RunSimuOnData(r[0],-1,int(nb_lignes/2),nb_arm)
        r[0]['Information sur la simulation'] = info
        status_text.text(f"{i+1} sur {len(Result)}")
        progress_bar.progress((i+1)/len(Result))

    Datas_simu = app_func.ordonner(Result)

############################# Affichage des resultats ####################################################

    # Configuration du Sommaire
    st.sidebar.title("Sommaire")

    i = 1
    for d in Datas_simu:
        tmp = d[1]
        st.sidebar.markdown("- ["+tmp+"](#section-"+str(i)+")")
        st.markdown("<a name='section-"+str(i)+"'></a>", unsafe_allow_html=True)#
        st.markdown(f"## {tmp}")
        app_func.display_dict(d[0],d[1])
        i=i+1

    st.sidebar.markdown("---")
    st.sidebar.markdown("Cliquez sur les liens ci-dessus pour accéder aux sections correspondantes.")

    if st.button("Répéter la simulation ?"):
        st.experimental_rerun()





