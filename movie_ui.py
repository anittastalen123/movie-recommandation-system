import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ----------------- Movie Recommender Logic -----------------

# 1️⃣ Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# 2️⃣ Merge datasets
movies = movies.merge(credits, on='title')

# 3️⃣ Keep useful columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# 4️⃣ Handle missing overviews
movies['overview'] = movies['overview'].fillna('')

# 5️⃣ Split overview into words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# 6️⃣ Convert JSON-like columns
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# 7️⃣ Extract director
def get_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(get_director)

# 8️⃣ Keep first 3 cast members
def get_cast(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
        if len(L) == 3:
            break
    return L

movies['cast'] = movies['cast'].apply(get_cast)

# 9️⃣ Combine all info into "tags"
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# 🔟 Simplified DataFrame
new_df = movies[['movie_id', 'title', 'tags']]

# 1️⃣1️⃣ Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# 1️⃣2️⃣ Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(new_df['tags']).toarray()

# 1️⃣3️⃣ Compute similarity
similarity = cosine_similarity(vectors)

# 1️⃣4️⃣ Recommendation function
def recommend(movie):
    movie_index = new_df[new_df['title'].str.lower() == movie.lower()].index
    if len(movie_index) == 0:
        return None
    movie_index = movie_index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_titles = [new_df.iloc[i[0]].title for i in movies_list]
    return recommended_titles

# ----------------- Tkinter UI -----------------

root = tk.Tk()
root.title("Movie Recommender")
root.geometry("800x600")
root.resizable(False, False)

# Optional: Background Image
try:
    bg_image = Image.open(r"C:\Users\anitt\OneDrive\Desktop\movie_rec\bg.jpg")
    bg_image = bg_image.resize((800, 600), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
except:
    pass  # If no image, skip

# Heading
heading = tk.Label(root, text="FILMORA MOVIE RECOMMENDER", font=("Helvetica", 28, "bold"))
heading.pack(pady=50)

# Search bar
entry = tk.Entry(root, font=("Helvetica", 16), width=35)
entry.pack(pady=20)
entry.focus()

# Search button
def search_movie():
    query = entry.get().strip()
    recommended = recommend(query)
    if recommended:
        messagebox.showinfo("Similar Movies", "\n".join(recommended))
    else:
        messagebox.showwarning("Not Found", "Movie not found! Try another title.")

search_btn = tk.Button(root, text="Search", font=("Helvetica", 14), command=search_movie)
search_btn.pack()

root.mainloop()