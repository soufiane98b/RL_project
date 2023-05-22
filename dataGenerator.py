
import pandas as pd
import random
import itertools
from functools import reduce
import scipy.stats as stats
from kmodes.kmodes import KModes
import numpy as np

def choix_compo_features(nb_feature,nb_min_categorie,nb_max_categorie):
    
    dict_composition_features = dict() # contient le nombre de catégorie pour chaque feature 
    columns = []
    nombre_individu_distinct = 1
    for i in range(nb_feature):
        feature = 'feature'+str(i)
        dict_composition_features[feature] = random.randint(nb_min_categorie, nb_max_categorie)
        columns.append(feature)
        nombre_individu_distinct = nombre_individu_distinct * dict_composition_features[feature]

    return dict_composition_features , nombre_individu_distinct

def generer_des_groupes(dict_composition_features,nb_groupes):
    groupes = []
    for i in range(nb_groupes):
        groupe_i = []
        for feature in dict_composition_features:
            max = dict_composition_features[feature]
            nb = random.randint(1,max)
            groupe_i.append(random.sample(range(1,max+1), nb))
        groupes.append(groupe_i)
    return groupes

def nb_possibilite(groupes):
    return [reduce((lambda x, y: x * y), list(map((lambda x: len(x) ), g))) for g in groupes]

def distance(x,y):
    set1 = set(x)
    set2 = set(y)
    similarity = len(set1.intersection(set2)) / len(set1.union(set2))
    return 1 - similarity

def distance_groupes(x,y):
    l = [distance(x[i],y[i]) for i in range(len(x)) ]
    return sum(l) / len(l)

def distance_moyenne_entre_paires_groupes(groupes):
    cmpt = 0
    summ = 0
    list_dist = []
    for pair in itertools.combinations(list(range(len(groupes))), 2):
        cmpt = cmpt + 1
        tmp = distance_groupes(groupes[pair[0]],groupes[pair[1]])
        summ = summ + tmp
        list_dist.append((pair[0],pair[1],tmp))
    return summ/cmpt , list_dist


def appartenance_groupe(individu,groupes):
    ind=0
    indices = []
    for g in groupes:
        test = True
        for i in range(len(g)):
            if individu[i] not in g[i]:
                test = False
                break
        if test == True:
            indices.append(ind)
        ind=ind+1
    return indices


def nombre_individu_par_groupes(df,groupes):
    cmpt_gr = [0] * len(groupes)
    intersection_cmpt = [0] * len(groupes)
    sans = 0
    for i in range(len(df)):
        individu = list(df.iloc[i])
        L = appartenance_groupe(individu,groupes)
        if L == [] :
            sans = sans + 1
        else :
            intersection_cmpt[len(L)-1] = intersection_cmpt[len(L)-1] + 1
        for l in L :
            cmpt_gr[l] = cmpt_gr[l] + 1
    return cmpt_gr , intersection_cmpt, sans

def predict_nombre_individu_par_groupes_sum(taille_df,nombre_individu_distinct,nb_possibilite):
    tmp = nb_possibilite / nombre_individu_distinct
    return tmp * taille_df


def choisir_alea(groupe,nb):
    max_unique_combi = nb_possibilite([groupe])[0]
    num_combos = min (max_unique_combi , nb)
    combos = []
    while len(combos) < num_combos :
        combo = []
        for g in groupe :
            combo.append(random.sample(g, 1)[0])
        if combo not in combos :
            combos.append(combo)
    return combos


def generer_all_unique_possi_df(groupe, id_groupe,taille):
    columns = ['feature'+str(f) for f in range(len(groupe))]
    combinations  = choisir_alea(groupe,taille)
    df = pd.DataFrame(combinations,columns=columns)
    df = df.assign(id_groupe = id_groupe)
    return df

def generer_all_unique_possi_df_for_groups(groupes,list_id,taille) :
    columns = ['feature'+str(f) for f in range(len(groupes[0]))]
    columns.append('id_groupe')
    df = pd.DataFrame(columns=columns)
    for i in list_id :
        tmp_df = generer_all_unique_possi_df(groupes[i], i,taille)
        df = pd.concat([df, tmp_df], axis=0)
    return df.reset_index().drop('index', axis=1)

def reblance_df(df,nb_groupes):
    # Et Avoir aussi au moins 10 rows, ca sera notre base pour entrainer le modèle 
    max_individu_grp = max(df['id_groupe'].value_counts())
    for id in range(nb_groupes) :
        nb_individu = (df['id_groupe']==id).sum()
        if(nb_individu<max_individu_grp):
            duppli =(max_individu_grp // nb_individu) - 1
            duppli_rand = max_individu_grp % nb_individu
            tmp_df = df[df['id_groupe']==id]
            if duppli > 0 :
                tmp_df = pd.concat([tmp_df] * duppli, ignore_index=True)
                tmp_df = pd.concat([tmp_df,tmp_df.sample(n=duppli_rand)], ignore_index=True)
            if duppli == 0:
                tmp_df = tmp_df.sample(n=duppli_rand)
            df = pd.concat([df,tmp_df], ignore_index=True)
    return df


def shape_train_df(df,nb_rows,len_group,ratio_balanced,dist='uniform'):
    distributions = {"expon": stats.expon(loc=0, scale=1),"poisson": stats.poisson(mu=1),"gamma": stats.gamma(a=1, loc=0, scale=1),"pareto": stats.pareto(b=1),"lognorm": stats.lognorm(s=1, loc=0, scale=1),"uniform": stats.uniform(loc=0, scale=1)}
    if dist not in distributions.keys():
        dist = 'uniform'
    # Le ratio commun entre 0 et 1 est départagé equitablement entre les différents groupes puis le 1 - ratio_commun est departagé inequitablement pour le desequilibre
    if (ratio_balanced<0 or ratio_balanced>1 ):
        print("Ratio not between 0 and 1")
        return None 
    base_ratios = [ratio_balanced/len_group for i in range(len_group)]
    unblanced_ratio = 1 - ratio_balanced
    dist_ratios = distributions[dist].rvs(size=len_group)
    dist_ratios = dist_ratios / sum(dist_ratios)
    dist_ratios = sorted(dist_ratios, reverse=True)
    dist_ratios = [r * unblanced_ratio for r in dist_ratios]
    wanted_groups_number = [int((x + y)*nb_rows) for x, y in zip(base_ratios, dist_ratios)] 
    random.shuffle(wanted_groups_number) 
    actual_groups_number = df['id_groupe'].value_counts()[0] 

    final_df = pd.DataFrame(columns=df.columns)

    for id_g  in range(len_group):
        wanted = wanted_groups_number[id_g]
        tmp_df = df[df['id_groupe']==id_g]
        if(wanted <= actual_groups_number):
            final_df = pd.concat([final_df,tmp_df.sample(n=wanted)],ignore_index=True)
        else :
            duppli =(wanted // actual_groups_number)
            duppli_rand = wanted % actual_groups_number
            tmp_df = pd.concat([tmp_df] * duppli, ignore_index=True)
            tmp_df = pd.concat([tmp_df,tmp_df.sample(n=duppli_rand)], ignore_index=True)
            final_df = pd.concat([final_df,tmp_df], ignore_index=True)
    
    return final_df 


def compter_diversite(df,nombre_groupe):
    disp = []
    for n in range(nombre_groupe) :
        tmp_df = df[df['id_groupe']==n]
        tmp_df = tmp_df[tmp_df.columns[:-1]]
        prop = tmp_df.duplicated().sum() * 100 / len(tmp_df)
        if np.isnan(prop):
            prop =0
        disp.append((n,str(int(prop))+'% duplique',len(tmp_df) ) )
    return disp



def ordonner(Datas):
    return sorted(Datas, key=lambda x: int(x[1][-2:])) # ATTENTION A CHANGER CAR MARCHE SI GROUPES SUR 2 CHIFFRE SINON ERREUR !!!!!

def generete_data(nombre_feature,nombre_div_cat_min,nombre_div_cat_max,nombre_groupe,nb_lignes,ratio_de_disparite_inter_group,dist):
    dict_info_data = {}
    ############################
    dict_info_data['nombre de ligne']=nb_lignes
    dict_info_data['ratio de disparite inter group'] = ratio_de_disparite_inter_group
    dict_info_data['Loi de distribution de dispartibe'] = dist
    ############################
    dict_composition_features , nombre_individu_distinct = choix_compo_features(nombre_feature,nombre_div_cat_min,nombre_div_cat_max)
    dict_info_data['nb de col avec one hot'] = sum(dict_composition_features.values())
    dict_info_data['indiviud distinct'] = nombre_individu_distinct
    dict_info_data['composition des features'] = dict_composition_features
    ############################
    groupes = generer_des_groupes(dict_composition_features,nombre_groupe)
    dict_info_data['composition des groupes'] = groupes
    moy , list_dist = distance_moyenne_entre_paires_groupes(groupes)
    nb_possi = nb_possibilite(groupes)
    dict_info_data['distance moyenne entre paires groupes'] = moy
    dict_info_data['nombre possibilite individu dans les groupes'] = nb_possi
    #dict_info_data['distance entre paires groupes '] = list_dist
    ############################
    test = True
    while test :
        df = generer_all_unique_possi_df_for_groups(groupes,list(range(len(groupes))),20000) # ->>>>> A VARIER SI BESOIN SI BCP BCP DE LIGNES
        df = df.sample(frac=1).reset_index(drop=True)
        df.drop_duplicates(subset=list(df.columns[:-1]), keep='last', inplace=True) # <----------- Enleve deux individu pareils pour ne pas etre dans 2 groupes differents
        # Car donne de mauvais resultats pour les algos voir si cas possible dans la realite , se produit quand très peu de nb individu discints 
        if len(df['id_groupe'].unique()) == len(groupes) : # ca veut dire un groupe inclu dans un autre et on veut pas ca , si egal ok
            test = False
            # Alors ok on peut passer étape suivante
    ############################
    df = reblance_df(df,len(groupes))
    final_df = shape_train_df(df,nb_lignes,len(groupes),ratio_de_disparite_inter_group,dist)
    dict_info_data['diversite des groupes'] = compter_diversite(final_df,nombre_groupe)
    ############################
    cmpt_gr , intersection_cmpt, sans= nombre_individu_par_groupes(final_df,groupes)
    dict_info_data['nb individu ds chaque groupes'] = cmpt_gr
    dict_info_data['nb de cross'] = intersection_cmpt
    ############################
    df_encoded = pd.get_dummies(final_df,columns=list(final_df.columns)[:-1]).astype(int)
    df_encoded['id_groupe'] = df_encoded.pop('id_groupe') # Déplacement en dernière ligne
    dict_info_data['nombre d individu duplique en % dans final df'] =(df_encoded.duplicated().sum())*100/len(df_encoded)
        
    if  (df_encoded.shape[1] != sum(dict_composition_features.values()) + 1):
        print('Il y aura probleme de noramlisation dans méthode lin ucb bis dans dict_composition_features')

    return final_df , df_encoded , dict_info_data

def reshape_dictionnaire_data(data):
    f=pd.DataFrame(data['other_info']['composition des features'],index=['nb of class'])
    ##################################################################################################
    ind = ['compo groupe '+str(i) for i in range(len(data['other_info']['composition des groupes']))]
    df = pd.DataFrame(data['other_info']['composition des groupes'],columns=f.columns,index =ind )
    ##################################################################################################
    general = pd.concat([f,df])
    general['nb disctint person possible'] = ['-']+data['other_info']['nombre possibilite individu dans les groupes']
    general['diversite des groupes'] = ['-']+data['other_info']['diversite des groupes']
    data['Compostion du Data Frame de Simulation'] = general
    tmp = data['other_info']
    del data['other_info']
    del tmp['composition des features']
    del tmp['composition des groupes']
    del tmp['nombre possibilite individu dans les groupes']
    del tmp['diversite des groupes']
    data['Autres Informations']= tmp
    return data

def cluster_data(dt,nb_group):
    data = dt
    km = KModes(n_clusters=nb_group, init='Huang', n_init=5) # MODELE DE CLUSTERING 
    km.fit(data.drop('id_groupe',axis=1)) 
    tmp_df = pd.DataFrame(pd.Series(km.labels_, name='Cluster'))
    data_clustered_encoded=pd.get_dummies(tmp_df,columns=['Cluster']).astype(int)
    data_clustered_encoded['id_groupe'] = data['id_groupe']
    count_cluster = pd.DataFrame(tmp_df['Cluster'].value_counts()).sort_values('Cluster').T
    return data_clustered_encoded , count_cluster

# MAIN FUNCTION 

def generator_multiple_df(L_col,L_cat,L_gps,nb_lines,ratio_desequilibre=1,dist='uniform',nb_cluster=5):
    avancement = len(L_col)*len(L_cat)*len(L_gps)
    i=1
    for c in L_col:
        for f in L_cat :
            for g in L_gps:
                final_df , df_encoded , dict_info_data = generete_data(c,2,f,g,nb_lines,ratio_desequilibre,dist)
                data_clustered_encoded , count_cluster = cluster_data(final_df,nb_cluster)
                # SHUFFLE 
                final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
                df_encoded = df_encoded.sample(frac=1, random_state=42).reset_index(drop=True)
                data_clustered_encoded = data_clustered_encoded.sample(frac=1, random_state=42).reset_index(drop=True)
                #####
                data = {'Nombre de colonne':c,
                        'Nombre de categorie max par colonne':f,
                        'Nombre de groupe':g,
                        'Data Frame de Simulation': final_df, 
                        'Data Frame one Hot Encodé':df_encoded,
                        'Data Frame clusterisé et one Hot Encodé' : data_clustered_encoded,
                        "Nombre d'occurence de chaque cluster":count_cluster,
                        'other_info': dict_info_data
                        }

                name = "Simulation du _"+str(c)+'_'+str(f)+'_'+str(g)
                yield (i ,avancement,reshape_dictionnaire_data(data),name) 
                i=i+1
    yield -1

