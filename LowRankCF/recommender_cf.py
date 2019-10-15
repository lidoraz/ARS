# Recommender




class Recommender():
    def __init__(self, rating_matrix, movies):
        self.rating_matrix = rating_matrix
        self.movies = movies

    def recommend_movie(self, user_inp):
        try:
            # user_inp=input('Enter the reference movie title based on which recommendations are to be made: ')
            user_inp = "Speed (1994)"
            inp = self.movies[self.movies['title'] == user_inp].index.tolist()
            inp = inp[0]

            self.movies['similarity'] = self.ratings_matrix.iloc[inp]
            self.movies.columns = ['movie_id', 'title', 'release_date', 'similarity']
            self.movies.head(5)

        except:
            print("Sorry, the movie is not in the database!")

        print("Recommended movies based on your choice of ", user_inp, ": \n",
              self.movies.sort_values(["similarity"], ascending=False)[1:10])