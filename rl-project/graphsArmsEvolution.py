
import matplotlib.pyplot as plt
import numpy as np

def initiate_ucb_ev_dict(listt):
    result = {}
    for i in listt :
        result[i]={'p':[],'theta':[],'Incert':[]}
    return result


def evolution_paramètre_arm(arm_id,listt,linucb_policy_object):
    arm_ucb_ev=linucb_policy_object.linucb_arms[arm_id].ucb_evo
    result=initiate_ucb_ev_dict( listt)
    for e in arm_ucb_ev:
        feature = e[0]
        result[feature]['p'].append(float(e[1]))
        result[feature]['theta'].append(float(e[2]))
        result[feature]['Incert'].append(float(e[3]))
    return result

# FOR EVOLUTION OF ARM : THETA , P and INCERTITUDE OF POLICY

def Evolution_Arm(arm,df_encoded,linucb_policy_object,fromm=0,to=-1):
    arm = 0
    fromm = 0
    to = -1
    listt = list(df_encoded['cluster'].value_counts().index)
    result=evolution_paramètre_arm(arm,listt,linucb_policy_object)
    for i in listt:
        plt.plot(result[i]['Incert'][fromm:to], label='theta of group'+str(i))
    plt.title("Evolution du Incert pour chaque feature de l'arm 0 ")
    plt.legend()
    plt.show()

    for i in listt:
        plt.plot(result[i]['theta'][fromm:to], label='theta of group'+str(i))
    plt.title("Evolution du theta pour chaque feature de l'arm 0 ")
    #plt.legend()
    plt.show()

    for i in listt:
        plt.plot(result[i]['p'][fromm:to], label='theta of group'+str(i))
    plt.title("Evolution du p pour chaque feature de l'arm 0 ")
    #plt.legend()
    plt.show()

# FOR EVOLUTION OF FEATURES ARM :  THETA , A and B FOR A POLICY

def Evoluion_feature_arm(arm,linucb_policy_object,fromm=0,to=-1):
    # Créer l'array de données
    arm = 0
    fromm = 0
    to = -1
    axe=0
    dataT = np.array(linucb_policy_object.linucb_arms[arm].theta_list[fromm:to])
    dataB = np.array(linucb_policy_object.linucb_arms[arm].b_list[fromm:to])
    dataA = np.array(linucb_policy_object.linucb_arms[arm].A_theta_list[fromm:to])

    # Afficher l'évolution de chaque point sur une courbe
    for i in range(dataT.shape[1]):
        plt.plot(dataT[:,i,:], label='feature {}'.format(i))

    plt.legend()
    plt.axvline(x=axe, color='black')
    plt.title("EVOLUTION DU THETA")
    plt.show()

    # Afficher l'évolution de chaque point sur une courbe
    for i in range(dataB.shape[1]):
        plt.plot(dataB[:,i,:], label='feature {}'.format(i))
    #----
    #plt.legend()
    #plt.axvline(x=axe, color='black')
    plt.title("EVOLUTION DU B")
    plt.show()

    # Afficher l'évolution de chaque point sur une courbe
    for i in range(dataA.shape[1]):
        plt.plot(dataA[:,i,:], label='feature {}'.format(i))
    #----
    #plt.legend()
    #plt.axvline(x=axe, color='black')
    plt.title("EVOLUTION DU A")
    plt.show()