import pandas as pd
import numpy as np
import math
from random import sample

def sigmoid(x):
    return (1 / (1 + math.exp(-x))) * 5.5


if __name__ == "__main__":
    # Read the ratings csv
    ratings = pd.read_csv("netflix_data/ml-100k/u.data", sep="\t",header=None,names=["user", "movie", "rating", "timestamp"])
    # Read the movie title csv to make actual recommendations
    movies = pd.read_csv("netflix_data/ml-100k/u.item", sep="|", encoding='latin-1', usecols=(0,1), header=None, names=["index", "title"])

    # Sample random indices for the test set (e.g., 20% of data)
    test_indices = sample(range(len(ratings)), int(len(ratings) / 5))

    # Create test and train sets
    test = ratings.iloc[test_indices]
    training = ratings.drop(index=test_indices)
    

    #Create the embeddings for the users
    user_embeddings = pd.DataFrame(data=ratings["user"].unique(), columns=["user"])
    user_embeddings['embeddings'] = [1/20 * np.random.random_sample(20) - 0.025 for _ in range(len(user_embeddings))]

    #Create the embeddings for the movies
    movie_embeddings = pd.DataFrame(data=ratings["movie"].unique(), columns=["movie"])
    movie_embeddings["embeddings"] = [1/20 * np.random.random_sample(20) - 0.025 for _ in range(len(movie_embeddings))]

    num_epochs = 10
    learning_rate = 0.01
    regularization_term = 0.02

    for e in range(num_epochs):
        #iterate ove each row
        total_loss = 0
        for item in training.itertuples():
            movie_embedding = movie_embeddings.loc[movie_embeddings["movie"] == item[2],"embeddings"].item()
            user_embedding = user_embeddings.loc[user_embeddings["user"] == item[1],"embeddings"].item()

            prediction = sigmoid((movie_embedding[:19] @ user_embedding[:19] + movie_embedding[19:] + user_embedding[19:])[0])

            loss = (item[3] - prediction)
            total_loss+=loss**2

            for i, embed in enumerate(movie_embedding):
                if (i != 19):
                    movie_embedding[i]-= learning_rate * (-2*loss*user_embedding[i] + 2 * regularization_term * movie_embedding[i])
                else:
                    movie_embedding[i]-= learning_rate * (-2*loss + 2 * regularization_term * movie_embedding[i])

            user_embedding = user_embedding
            for i, embed in enumerate(movie_embedding):
                if (i != 19):
                    user_embedding[i] = learning_rate * (-2*loss*movie_embedding[i] + 2 * regularization_term * user_embedding[i])
                else:
                    user_embedding[i] = learning_rate * (-2*loss + 2 * regularization_term * user_embedding[i])  
        root_mean_squared_error = total_loss ** (1/2)
        print(f"RMSE in epoch {e+1}: {root_mean_squared_error}")

        total_off = 0
        for item in test.itertuples():
            movie_embedding = movie_embeddings.loc[movie_embeddings["movie"] == item[2],"embeddings"].item()
            user_embedding = user_embeddings.loc[user_embeddings["user"] == item[1],"embeddings"].item()
            prediction = sigmoid((movie_embedding[:19] @ user_embedding[:19] + movie_embedding[19:] + user_embedding[19:])[0])
            total_off+=abs(item[3]-prediction)
        print(f"Mean error is {total_off/len(test)}")





            


