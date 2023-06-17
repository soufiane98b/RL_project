import numpy as np
import math

class linucb_disjoint_arm():

        # each arm can have his own set of parameters (xi), this why we call it disjoint
    def __init__(self, arm_index, d, alpha,version):

        #Version of algorihtm
        self.version = version

        # Track arm index
        self.arm_index = arm_index
        
        # Keep track of alpha
        self.alpha = alpha
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A = np.identity(d)
        
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d,1])

        self.A_theta_list = []
        self.A_incertitude_list = []
        self.b_list = []
        self.theta_list = []
        self.ucb_evo = []
        self.alpha_list = []
                
    def calc_UCB(self, x_array):
        # Find A inverse for ridge regression
            
        self.A_inv = np.linalg.inv(self.A) 

        
        self.theta = np.dot(self.A_inv, self.b) # each point of covariate estime une importance globale qu'accorde une arm à chaque feature
        
        x = x_array.reshape([-1,1])
       
        theta_x = np.dot(self.theta.T,x)
        incert_x =  self.alpha * np.sqrt(np.dot(x.T, np.dot(self.A_inv,x)))

        p = theta_x +   incert_x # !!TESTER AVEC DEUX X DIFFERENTS CE QUE CA DONNE!!
        
        # la récompense la plus haute espéré pour ce bras
        # et on rajouter la std pour l'exploration des arm qui ont une grande incertitude dans la variance de leur features 
        """self.A_theta_list.append(self.A.copy())
        self.A_incertitude_list.append(self.A.copy())
        self.b_list.append(self.b.copy())
        self.theta_list.append(self.theta.copy())
        self.alpha_list.append(self.alpha)"""
        # -----------------
        return p , theta_x , incert_x
    
    
    def reward_update(self, reward, x_array):
        x = x_array.reshape([-1,1])
        self.A = np.add(self.A, np.dot(x, x.T))
        self.b = np.add(self.b, reward*x)
    

class linucb_policy():
    
    def __init__(self, K_arms, d, alpha,version,df_encoded):
        self.df_encoded = df_encoded
        self.K_arms = K_arms
        self.version = version
        self.linucb_arms = [linucb_disjoint_arm(arm_index = i, d = d, alpha = alpha,version = version) for i in range(K_arms)]
    
        
    def select_arm(self, x_array):
        # Initiate ucb to be 0
        highest_ucb = -10
        
        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []
        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb , theta_x , incert_x = self.linucb_arms[arm_index].calc_UCB(x_array)

            ##### pour suivre l'evoulution de ucb et theta du group
            """indice = np.where((self.df_encoded.iloc[:, :self.df_encoded.shape[1]-1] == x_array).all(axis=1))[0][0] ##### @@@@@@ @@@@@@ @@@@@@ @@@@@@  
            indice = self.df_encoded.iloc[indice][-1] ##### @@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ 
            self.linucb_arms[arm_index].ucb_evo.append((indice,arm_ucb[0][0],theta_x[0][0],incert_x[0][0]))"""
            #####
            
            # If current arm is highest than current highest_ucb
            if arm_ucb > highest_ucb:
                
                # Set new max ucb
                highest_ucb = arm_ucb
                
                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [(arm_index,arm_ucb)]

            # If there is a tie, append to candidate_arms
            elif arm_ucb == highest_ucb:
                candidate_arms.append((arm_index,arm_ucb))
        # Choose based on candidate_arms randomly (tie breaker)
        if candidate_arms==[]:
            print('PBBB')
            print(highest_ucb)
            print(candidate_arms)
        chosen_arm = [t[0] for t in candidate_arms]
        chosen_arm = np.random.choice(chosen_arm)
        
        return chosen_arm
    
########################################################################################################################

def redimensionner(A, b, precision=2):
    div = np.ceil(b[:, 0] / A[:, 0] * 10 ** precision) / 10 ** precision
    mask = (A[:, 0] > 10 ** (precision + 1)) & (div > 1 / 10 ** precision) # Very important precision+1 to avoid stagnation 
    A[mask, 0] = 10 ** precision
    b[mask, 0] = div[mask] * 10 ** precision
    return A, b


def indice_norm(dict_composition_features):
    l = []
    cmpt = 0
    for v in dict_composition_features.values():
        l.append((cmpt,v+cmpt))
        cmpt = cmpt + v
    return l


def normaliser(theta):
    return ((theta / (np.sum(theta)+0.0000000001)) * 2) - 1


def normaliser_2(theta,dict_composition_features):
    if dict_composition_features == -1:
        return theta
    L = indice_norm(dict_composition_features)
    final_theta = []
    for i in L:
        final_theta.append(normaliser(theta[i[0]:i[1]]))
    final_theta = np.concatenate(final_theta) 
    return final_theta / len(dict_composition_features)

####################################################        BIS       #####################################################


class linucb_disjoint_arm_bis():

    # each arm can have his own set of parameters (xi), this why we call it disjoint
    def __init__(self, arm_index, d, alpha,version,compo_feature):

        #Version of algorihtm
        self.version = version

        # Track arm index
        self.arm_index = arm_index
        
        # Keep track of alpha
        self.alpha = alpha
        
        # A: (d x d) matrix = D_a.T * D_a + I_d. 
        # The inverse of A is used in ridge regression 
        self.A_incertitude = np.ones((d, 1))
        self.A_theta = np.ones((d, 1))
        
        # b: (d x 1) corresponding response vector. 
        # Equals to D_a.T * c_a in ridge regression formulation
        self.b = np.zeros([d,1])

        # Composition of feature for normalisation
        self.compo_feature= compo_feature

        # For Interpretation 
        self.A_theta_list = []
        self.A_incertitude_list = []
        self.b_list = []
        self.theta_list = []
        self.ucb_evo = []
        self.alpha_list = []
                
    def calc_UCB(self, x_array):

            
        self.A_inv_incertitude = np.reciprocal(self.A_incertitude )
        self.theta = np.divide(self.b,self.A_theta)


        x = x_array.reshape([-1,1])

        

        incertitude = np.sqrt(np.dot( (self.A_inv_incertitude).T,x)[0][0])

        theta = np.dot(normaliser_2(self.theta,self.compo_feature).T,x)[0][0]

        p = theta +   self.alpha * incertitude
        
        # la récompense la plus haute espéré pour ce bras
        # et on rajouter la std pour l'exploration des arm qui ont une grande incertitude dans la variance de leur features 
        """self.A_theta_list.append(self.A_theta.copy())
        self.A_incertitude_list.append(self.A_incertitude.copy())
        self.b_list.append(self.b.copy())
        self.theta_list.append(normaliser_2(self.theta,self.compo_feature))
        self.alpha_list.append(self.alpha)"""
        # -----------------
        
        return p , theta , self.alpha * incertitude
     
    def reward_update(self, reward, x_array):
        # Reshape covariates input into (d x 1) shape vector        
        x = x_array.reshape([-1,1])
        self.A_incertitude = np.add(self.A_incertitude, x)
        self.A_theta = np.add(self.A_theta, x)
        self.b = np.add(self.b, reward*x)

        if self.version == 1 :
            self.A_theta,self.b = redimensionner(self.A_theta,self.b)

class linucb_policy_bis():
    
    def __init__(self, K_arms, d, alpha,version,compo_feature,df_encoded):
        self.df_encoded = df_encoded
        self.K_arms = K_arms
        self.version = version
        self.compo_feature = compo_feature
        self.linucb_arms = [linucb_disjoint_arm_bis(arm_index = i, d = d, alpha = alpha,version = version,compo_feature = compo_feature ) for i in range(K_arms)]
    
        
    def select_arm(self, x_array):
        # Initiate ucb to be 0
        highest_ucb = -10
        
        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []
        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb , theta , incert = self.linucb_arms[arm_index].calc_UCB(x_array)

            ##### pour suivre l'evoulution de ucb et theta du group
            """indice = np.where((self.df_encoded.iloc[:, :self.df_encoded.shape[1]-1] == x_array).all(axis=1))[0][0] ##### @@@@@@ @@@@@@ @@@@@@ @@@@@@  
            indice = self.df_encoded.iloc[indice][-1] ##### @@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ @@@@@@ 
            self.linucb_arms[arm_index].ucb_evo.append((indice,arm_ucb,theta,incert))"""
            #####

            # If current arm is highest than current highest_ucb
            if arm_ucb > highest_ucb:
                
                # Set new max ucb
                highest_ucb = arm_ucb
                
                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [(arm_index,arm_ucb)]

            # If there is a tie, append to candidate_arms
            elif arm_ucb == highest_ucb:
                candidate_arms.append((arm_index,arm_ucb))
        # Choose based on candidate_arms randomly (tie breaker)
        if candidate_arms==[]:
            print('PBBB')
            print(highest_ucb)
            print(candidate_arms)
        chosen_arm = [t[0] for t in candidate_arms]
        chosen_arm = np.random.choice(chosen_arm)
        
        return chosen_arm
    
########################################################################################################################




