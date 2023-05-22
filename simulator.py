import random
import numpy as np
import pandas as pd
import linucb
import time


def simulate_reward(cluster_id,arm_id,nb_arm):
    penality = 0
    if cluster_id%nb_arm == arm_id: # in this case we choosed the good arm but doesn't mean for sure postive reward but high proba
        p = 0.5 + round(random.uniform(-0.2, 0.2), 1)
        return random.choices([penality,1], [1-p,p])[0]
    else :
        p = 0.1 + round(random.uniform(-0.05, 0.05), 2)
        return random.choices([penality,1], [1-p,p])[0]

def simulate_reward_shift(cluster_id,arm_id,nb_gps,nb_arm):
    prefered_arm =  shift(cluster_id,nb_gps)
    penality = 0
    if prefered_arm%nb_arm == arm_id: # in this case we choosed the good arm but doesn't mean for sure postive reward but high proba
        p = 0.6 + random.uniform(-0.1, 0.2)
        return random.choices([penality,1], [1-p,p])[0]
    else :
        p = 0.1 + random.uniform(-0.05, 0.05)
        return random.choices([penality,1], [1-p,p])[0]
    

def shift(c_id,nb_gps):
    t = (nb_gps-1) - c_id
    if (t<0 or t>nb_gps-1):
        print("nb_gps-1 : ",nb_gps-1,"c_id",c_id," t:",t)
        print('erreur dans shift')
    return t

####################################################################################################################################

def simulator(i,df,df_clustered,nb_groups,policies,shift,nb_arms,arm_index,data_reward,aligned_time,aligned_time_steps,cumulative_rewards,aligned_ctr):
    for p in range(len(policies)):
            # We calculate the time for each iteration  
            start_time = time.time()
            ##########
            if p==2: # CAS ALEATOIRE
                # Recupere Data
                array = np.array(df.iloc[i])
                data_x_array = np.delete(array,-1) # enlève dernier élement 
                ######################
                arm_index[p] = random.randint(0, nb_arms-1) # WE HAVE 10 ARMS
                if shift!=-1:
                     data_reward[p] = simulate_reward(array[-1],arm_index[p],nb_arms)
                else :
                     data_reward[p] = simulate_reward_shift(array[-1],arm_index[p],nb_groups,nb_arms)
            else :
                if p == 3:  # CAS DATA CLUSTERISE
                    # Recupere Data
                    array = np.array(df_clustered.iloc[i])
                    data_x_array = np.delete(array,-1) # enlève dernier élement 
                else : # TOUT AUTRE CAS
                    # Recupere Data
                    array = np.array(df.iloc[i])
                    data_x_array = np.delete(array,-1) # enlève dernier élement 

                arm_index[p] = policies[p].select_arm(data_x_array)
                if shift !=-1:
                     data_reward[p] = simulate_reward(array[-1],arm_index[p],nb_arms)
                else :
                     data_reward[p] = simulate_reward_shift(array[-1],arm_index[p],nb_groups,nb_arms)
                # Use reward information for the chosen arm to update
                tmp_arm = int(arm_index[p])
                policies[p].linucb_arms[tmp_arm].reward_update(data_reward[p], data_x_array)
                ##########
            end_time = time.time()
            elapsed_time = end_time - start_time
            aligned_time[p]  = aligned_time[p] + elapsed_time
            # For CTR calculation
            aligned_time_steps[p] +=1
            cumulative_rewards[p] += data_reward[p]
            aligned_ctr[p].append(cumulative_rewards[p]/aligned_time_steps[p]) 


def ctr_simulator_policies(df,df_clustered,nb_groups,policies,shift,nb_arms):
    # Instantiate trackers
    cumulative_rewards_b_a = []
    n_policies = len(policies)
    aligned_time_steps = np.zeros(n_policies)
    cumulative_rewards = np.zeros(n_policies)
    aligned_ctr = [[] for i in range(n_policies)]
    aligned_time = np.zeros(n_policies)
    # For updating arm indexes
    arm_index = np.zeros(n_policies)
    # For updating arm rewards
    data_reward = np.zeros(n_policies)
    for i in range(len(df)):
        simulator(i,df,df_clustered,nb_groups,policies,-1,nb_arms,arm_index,data_reward,aligned_time,aligned_time_steps,cumulative_rewards,aligned_ctr)    
    cumulative_rewards_b_a.append(cumulative_rewards)
    ##################################################### SHIFT #####################################################
    cumulative_rewards = np.zeros(n_policies)
    aligned_time_steps = np.zeros(n_policies)
    for i in range(shift):
        simulator(i,df,df_clustered,nb_groups,policies,shift,nb_arms,arm_index,data_reward,aligned_time,aligned_time_steps,cumulative_rewards,aligned_ctr)
    cumulative_rewards_b_a.append(cumulative_rewards)

    return (cumulative_rewards_b_a, aligned_ctr ,aligned_time,policies)

##########################################################   MAIN   ################################################################

def export_simu_info(cumulative_rewards, aligned_ctr ,aligned_time ):
    info_simu = dict()
    ########
    info_simu['Aligned CTR'] = [(aligned_ctr[0],'Linear UCB'),(aligned_ctr[1],'Linear UCB BIS'),(aligned_ctr[2],'Random'),(aligned_ctr[3],'BIS Clustering')]
    ########
    df = pd.DataFrame(aligned_time,index=['Linear UCB Disjoint','Linear UCB Disjoint BIS','Random','BIS Clustering'],columns=["Execution time"]).T
    tmp = pd.DataFrame(cumulative_rewards[0],index=['Linear UCB Disjoint','Linear UCB Disjoint BIS','Random','BIS Clustering'],columns=["Cumulative Rewards Before Shift"]).T
    df = pd.concat([df , tmp])
    tmp = pd.DataFrame(cumulative_rewards[1],index=['Linear UCB Disjoint','Linear UCB Disjoint BIS','Random','BIS Clustering'],columns=["Cumulative Rewards After Shift"]).T
    df = pd.concat([df , tmp])
    tmp = pd.DataFrame(sum(cumulative_rewards),index=['Linear UCB Disjoint','Linear UCB Disjoint BIS','Random','BIS Clustering'],columns=["Cumulative Rewards Total"]).T
    df = pd.concat([df , tmp])
    info_simu['Information sur Reward et execution'] = df
    return info_simu

# MAIN
def RunSimuOnData(Data,l,l_shift,nb_arms):
    df_encoded = Data['Data Frame one Hot Encodé']
    nb_groups = Data['Nombre de groupe']
    df_clustered_encoded = Data['Data Frame clusterisé et one Hot Encodé']
    if l != -1 : # Sinon on parcout tout le data frame
        df_encoded = df_encoded.head(l)
    dict_info_data = dict(Data['Compostion du Data Frame de Simulation'].loc['nb of class'])
    del dict_info_data['nb disctint person possible']
    del dict_info_data['diversite des groupes']

    # Choice of policies 
    L_UCB = linucb.linucb_policy(K_arms = nb_arms, d = df_encoded.shape[1]-1, alpha=1,version= -1,df_encoded=df_encoded)
    L_UCB_BIS = linucb.linucb_policy_bis(K_arms = nb_arms, d = df_encoded.shape[1]-1, alpha=0.5,version= 1,compo_feature=dict_info_data,df_encoded=df_encoded)
    L_UCB_BIS_clustered = linucb.linucb_policy_bis(K_arms = nb_arms, d = df_clustered_encoded.shape[1]-1, alpha=1,version= 1,compo_feature=-1,df_encoded=df_clustered_encoded)
    policies = [L_UCB,L_UCB_BIS,'Random',L_UCB_BIS_clustered]
   
    # Run the Simulation and recover data of simu
    cumulative_rewards, aligned_ctr ,aligned_time ,policies = ctr_simulator_policies(df_encoded,df_clustered_encoded,nb_groups,policies,l_shift,nb_arms)
    
    return export_simu_info(cumulative_rewards, aligned_ctr ,aligned_time )


