import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_top_candidates(job_description, top_n=5,df):
    if df.empty:
        return []

    # Combine JD with candidates
    all_text = df['Combined_Profile'].tolist() + [job_description]

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_text)

    # Calculate similarity
    jd_vector = tfidf_matrix[-1]
    candidate_vectors = tfidf_matrix[:-1]
    scores = cosine_similarity(jd_vector, candidate_vectors).flatten()

    # Get results
    results_df = df.copy()
    results_df['Match_Score'] = scores

    # Sort and convert to dictionary
    top_candidates = results_df.sort_values(by='Match_Score', ascending=False).head(top_n)
    return top_candidates.to_dict(orient='records')