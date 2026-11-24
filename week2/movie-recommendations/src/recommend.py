"""Movie recommendations using genres, tags, and user ratings."""

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("data/ml-latest-small")


def load_movielens():
    """Download and load MovieLens files if they are not already present."""
    required_files = ["movies.csv", "ratings.csv", "tags.csv"]
    if not all((DATA_DIR / filename).exists() for filename in required_files):
        print("Downloading MovieLens Latest Small...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        with urlopen(DATA_URL, timeout=60) as response:
            archive_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for filename in required_files:
                member = f"ml-latest-small/{filename}"
                with archive.open(member) as source:
                    (DATA_DIR / filename).write_bytes(source.read())

    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    return movies, ratings, tags


def make_content_matrix(movies, tags):
    """Create TF-IDF movie vectors from genres and user tags."""
    tag_text = (
        tags.groupby("movieId")["tag"]
        .apply(lambda values: " ".join(values.astype(str)))
        .rename("tag_text")
    )

    catalog = movies.merge(tag_text, on="movieId", how="left")
    catalog["tag_text"] = catalog["tag_text"].fillna("")
    catalog["content"] = (
        catalog["genres"].str.replace("|", " ", regex=False)
        + " "
        + catalog["tag_text"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    content_matrix = vectorizer.fit_transform(catalog["content"])
    return catalog, content_matrix


def find_movie(catalog, title_query):
    """Find a movie by exact title or case-insensitive partial match."""
    query = title_query.strip().lower()
    exact = catalog[catalog["title"].str.lower() == query]
    if not exact.empty:
        return int(exact.index[0])

    matches = catalog[
        catalog["title"].str.lower().str.contains(query, regex=False)
    ]
    if matches.empty:
        raise ValueError(f"No movie title found matching {title_query!r}.")
    return int(matches.index[0])


def recommend_similar_movie(catalog, content_matrix, title_query, limit):
    """Return movies with genres/tags most similar to a selected movie."""
    movie_index = find_movie(catalog, title_query)
    scores = cosine_similarity(
        content_matrix[movie_index],
        content_matrix,
    ).ravel()

    ranked = np.argsort(scores)[::-1]
    ranked = [idx for idx in ranked if idx != movie_index][:limit]

    print(f"\nMovies similar to: {catalog.iloc[movie_index]['title']}")
    for idx in ranked:
        print(f"{scores[idx]:.3f}  {catalog.iloc[idx]['title']}")


def recommend_content_for_user(
    catalog,
    content_matrix,
    ratings,
    user_id,
    limit,
):
    """Build a user taste profile from their higher-rated movie content."""
    user_ratings = ratings[ratings["userId"] == user_id]
    if user_ratings.empty:
        raise ValueError(f"User ID {user_id} has no ratings in this dataset.")

    # Favor movies rated above 3 stars when constructing the taste profile.
    liked = user_ratings[user_ratings["rating"] >= 3.5]
    if liked.empty:
        raise ValueError(
            f"User {user_id} has no ratings of 3.5 or higher to build a profile."
        )

    index_by_movie = pd.Series(catalog.index, index=catalog["movieId"])
    liked = liked[liked["movieId"].isin(index_by_movie.index)]

    movie_indices = index_by_movie.loc[liked["movieId"]].to_numpy()
    weights = (liked["rating"].to_numpy() - 3.0).reshape(-1, 1)
    profile = content_matrix[movie_indices].multiply(weights).sum(axis=0)

    scores = cosine_similarity(
        np.asarray(profile),
        content_matrix,
    ).ravel()

    rated_ids = set(user_ratings["movieId"])
    ranked = np.argsort(scores)[::-1]
    ranked = [
        idx for idx in ranked
        if catalog.iloc[idx]["movieId"] not in rated_ids
    ][:limit]

    print(f"\nContent-based recommendations for user {user_id}")
    for idx in ranked:
        print(f"{scores[idx]:.3f}  {catalog.iloc[idx]['title']}")


def recommend_collaborative(movies, ratings, user_id, limit):
    """Recommend unrated movies using similar users' rating patterns."""
    rating_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
    )

    if user_id not in rating_matrix.index:
        raise ValueError(f"User ID {user_id} has no ratings in this dataset.")

    user_means = rating_matrix.mean(axis=1)
    centered = rating_matrix.sub(user_means, axis=0).fillna(0)
    observed = rating_matrix.notna().astype(float)

    # Compare users by their centered movie-rating patterns.
    user_similarity = cosine_similarity(centered.to_numpy())
    user_ids = rating_matrix.index.to_list()
    target_index = user_ids.index(user_id)

    # Exclude the target user from their own neighbor recommendations.
    user_similarity[target_index, target_index] = 0
    user_similarity = np.maximum(user_similarity, 0)

    # Estimate a movie score from similar users who actually rated it.
    similarity_weights = user_similarity[target_index]
    weighted_deviations = similarity_weights @ centered.to_numpy()
    available_weights = similarity_weights @ observed.to_numpy()

    target_mean = user_means.loc[user_id]
    predicted_scores = np.full(len(rating_matrix.columns), target_mean)
    valid = available_weights > 0
    predicted_scores[valid] += (
        weighted_deviations[valid] / available_weights[valid]
    )

    already_rated = rating_matrix.loc[user_id].notna().to_numpy()
    predicted_scores[already_rated] = -np.inf

    ranked_columns = np.argsort(predicted_scores)[::-1]
    movie_lookup = movies.set_index("movieId")
    print(f"\nCollaborative recommendations for user {user_id}")

    shown = 0
    for column_index in ranked_columns:
        if not np.isfinite(predicted_scores[column_index]):
            continue
        movie_id = rating_matrix.columns[column_index]
        if movie_id not in movie_lookup.index:
            continue

        title = movie_lookup.loc[movie_id, "title"]
        print(f"Predicted rating {predicted_scores[column_index]:.2f}  {title}")
        shown += 1
        if shown >= limit:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="MovieLens user ID (default: 1).",
    )
    parser.add_argument(
        "--method",
        choices=["both", "content", "collaborative"],
        default="both",
    )
    parser.add_argument(
        "--similar-to",
        help="Movie title to find genre/tag similar movies for.",
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    movies, ratings, tags = load_movielens()
    catalog, content_matrix = make_content_matrix(movies, tags)

    if args.similar_to:
        recommend_similar_movie(
            catalog,
            content_matrix,
            args.similar_to,
            args.limit,
        )
        return

    if args.method in ("both", "content"):
        recommend_content_for_user(
            catalog,
            content_matrix,
            ratings,
            args.user_id,
            args.limit,
        )

    if args.method in ("both", "collaborative"):
        recommend_collaborative(
            movies,
            ratings,
            args.user_id,
            args.limit,
        )


if __name__ == "__main__":
    main()