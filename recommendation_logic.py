import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def normalize_skill(skill_name):
    """Maps common skill variations to the exact database tags."""
    synonym_map = {
        "data analytics": "Data Analysis",
        "analytics": "Data Analysis",
        "ml": "Machine Learning",
        "ai": "Machine Learning",
        "dl": "Deep Learning",
        "postgres": "PostgreSQL",
        "k8s": "Kubernetes",
        "py": "Python"
    }
    cleaned = skill_name.strip().lower()
    return synonym_map.get(cleaned, skill_name.strip())

def main():
    print("=== Project 3 ===")
    print("Task: Tech Stack & Career Recommender Engine\n")

    # Dataset with comma-separated skill tags
    career_dataset = pd.DataFrame({
        "job_role": [
            "Data Scientist",
            "DevOps Engineer",
            "Backend Developer",
            "Machine Learning Engineer",
            "Cloud Architect"
        ],
        "required_skills": [
            "Python, SQL, Machine Learning, Data Analysis, Statistics, Pandas",
            "AWS, Docker, Kubernetes, CI/CD, Linux, Cloud Automation",
            "Java, Python, SQL, APIs, Microservices, PostgreSQL, Django",
            "Python, TensorFlow, PyTorch, Machine Learning, Deep Learning, Computer Vision",
            "AWS, Azure, Cloud Architecture, Docker, Terraform, Networking"
        ]
    })

    # Extract clean unique skills for menu display
    all_skills = set()
    for skill_string in career_dataset["required_skills"]:
        all_skills.update([skill.strip() for skill in skill_string.split(",")])
    
    sorted_skills = sorted(list(all_skills))

    print("--- Available Skills in Database ---")
    print(", ".join(sorted_skills))
    print("-" * 50 + "\n")

    # Step 1: Ingestion & Skill Normalization
    print("--- Step 1: User Profile Ingestion ---")
    raw_input = input("Enter your skills from the list above (comma-separated): ")
    
    # Process and normalize raw input
    raw_list = [s.strip() for s in raw_input.split(",") if s.strip()]
    user_skills_input = [normalize_skill(skill) for skill in raw_list]
    
    if not user_skills_input:
        user_skills_input = ["Python", "Cloud Automation", "Docker"]
        print(f"No input detected. Defaulting to: {user_skills_input}")

    user_profile_text = " ".join(user_skills_input)
    print(f"Normalized Input Skills: {user_skills_input}\n")

    # Step 2: Feature Extraction using TF-IDF
    tfidf_encoder = TfidfVectorizer(token_pattern=r"(?u)\b[\w/]+\b")
    role_skill_vectors = tfidf_encoder.fit_transform(career_dataset["required_skills"])
    user_vector = tfidf_encoder.transform([user_profile_text])

    # Step 3: Scoring via Cosine Similarity
    similarity_scores = cosine_similarity(user_vector, role_skill_vectors).flatten()
    career_dataset["match_score"] = similarity_scores

    # Step 4: Sorting & Top-N Filtering
    sorted_recommendations = career_dataset.sort_values(by="match_score", ascending=False)
    top_recommendations = sorted_recommendations.head(3)

    print("--- Top 3 Recommended Career Paths ---")
    for position, (_, row) in enumerate(top_recommendations.iterrows(), start=1):
        match_percentage = row["match_score"] * 100
        print(f"{position}. {row['job_role']} — Match Score: {match_percentage:.1f}%")
        print(f"   Skills Required: {row['required_skills']}\n")

if __name__ == "__main__":
    main()