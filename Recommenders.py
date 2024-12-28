import pandas as pd
import numpy as np

#Define recommenders module
class popularity_recommender_py():
    def __init__(self):
        self.train_data=None
        self.user_id=None
        self.item_id=None
        self.popularity_recommendations=None
    
    def create(self, train_data, user_id, item_id):
        self.train_data=train_data
        self.user_id=user_id
        self.item_id=item_id
        #Count of user_ids for each unique song as rec score
        train_data_grouped=train_data.groupby([self.item_id]).agg({self.user_id:'count'}).reset_index()
        #Renaming Column user_id to score
        train_data_grouped.rename(columns = {'user_id': 'score'}, inplace=True)
        train_data_sort=train_data_grouped.sort_values(['score', self.item_id], ascending=[0, 1])
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        self.popularity_recommendations=train_data_sort.head(10) #Gets the top ten recommendations

    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations
        user_recommendations['user_id'] = user_id
        #Reordering columns for convenience
        cols = user_recommendations.columns.tolist()        
        cols = cols[-1:] + cols[:-1]        
        user_recommendations = user_recommendations[cols]
        return user_recommendations


#Class for Item similarity based recommender
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data=None
        self.user_id=None
        self.item_id=None
        self.cooccurence_matrix=None
        self.songs_dict=None
        self.rev_songs_dict=None
        self.item_similarity_recommendations=None

    #Get unique songs corresponding to a given user
    def get_user_items(self, user):
        user_data=self.train_data[self.train_data[self.user_id]==user]
        user_items=list(user_data[self.item_id].unique())
        return user_items
    
    #Get unique users for an item
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
        return item_users
    
    #Get unique songs in entire dataset
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
        return all_items
    
    #Constructing a cooccurence matrix for similarities
    #user_song_users is a list of sets where each set contains the users who interacted with a specific song in user_songs
    def construct_cooccurence_matrix(self, user_songs, all_songs):
        user_songs_users=[]
        #For each user that interacted with that song, retrieve the set of users who interacted with that song as well
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

        #Initialize a co-occurence matrix of size (len(user_songs)->row, len(all_songs)-> columns with zeros.
        cooccurence_matrix=np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        for i in range(0, len(all_songs)): #Column wise
            songs_i_data=self.train_data[self.train_data[self.item_id]==all_songs[i]] #Filter TRAINING data to only include rows corresp to current song
            users_i=set(songs_i_data[self.user_id].unique()) #from songs_i_data gets the set of unique users who interacted with this current song

            for j in range(0, len(user_songs)): #Columns lo rows chusthunnaamu
                users_j=user_songs_users[j]
                #get the intersection
                users_intersection=users_i.intersection(users_j)

                if len(users_intersection)!=0:
                    users_union=users_i.union(users_j)
                    cooccurence_matrix[j, i]= float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j, i]=0

        return cooccurence_matrix
    
    #Generating Recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix)) #Tells us how sparse the matrix is
        #Compute similarity scores
        user_sim_scores=cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0]) #Sum for each column (ante each song) divided by total number of user songs, to normalise
        user_sim_scores=np.array(user_sim_scores)[0].tolist() #Convert scores to list for convenience
        #Sort all songs by similarity scores in descending order
        sort_index=sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)
        #Initialize an empty DataFrame and populate it with Recommendations
        columns=['user_id', 'song', 'score', 'rank']
        df=pd.DataFrame(columns=columns)
        rank=1
        for i in range(0, len(sort_index)):
            #We're now going to check 3 conditions: score should not be NaN, songs should not already be in user's history and limiting rank to 10 only
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <=10:
                #Adding new row to Dataframe
                df.loc[len(df)]=[user, all_songs[sort_index[i][1]], sort_index[i][0], rank]
                rank=rank+1
        #Handle no recs
        if df.shape[0]==0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
    
    #Model creation and recommendation
    def create(self, train_data, user_id, item_id):
        self.train_data=train_data
        self.user_id=user_id
        self.item_id=item_id
    
    def recommend(self, user):
        user_songs=self.get_user_items(user)
        print("No. of unique songs for the user: %d" % len(user_songs))
        all_songs = self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))
        cooccurence_matrix=self.construct_cooccurence_matrix(user_songs, all_songs)
        df_recommendations=self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

        return df_recommendations
    
    #This basically gives you similar songs to the songs you input, instead of on the user's history
    def get_similar_items(self, item_list):
        user_songs=item_list
        all_songs=self.get_all_items_train_data()
        print("no. of unique songs in the training set: %d" % len(all_songs))

        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
     
        return df_recommendations