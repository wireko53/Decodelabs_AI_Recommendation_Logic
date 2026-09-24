# Tech Stack & Career Recommender Engine

Project 3 of my AI Internship at DecodeLabs.

## What I Built

A content-based recommendation engine that takes a user's skill set and suggests the job roles that match it most closely, using TF-IDF vectorization and cosine similarity — the same core technique behind many real-world recommender systems.

## Tech Stack

- Python
- pandas
- scikit-learn (TF-IDF vectorizer, cosine similarity)

## How It Works

1. **Display Available Skills:** The full list of valid skill tags in the database is printed up front, so the user knows what they can enter
2. **Ingest:** The user types their skills as a comma-separated list at a prompt. If nothing is entered, the engine falls back to a default profile (`Python`, `Cloud Automation`, `Docker`)
3. **Normalize:** Each entered skill is checked against a synonym map that catches common variations and abbreviations (e.g. `ml` and `ai` → `Machine Learning`, `k8s` → `Kubernetes`, `py` → `Python`, `postgres` → `PostgreSQL`) and converts them to the exact tags used in the database
4. **Vectorize:** A `TfidfVectorizer` is fit on each job role's required-skills text, weighting distinctive skills higher than generic ones
5. **Score:** The user's skill vector is compared against every job role's vector using cosine similarity
6. **Rank:** Roles are sorted by match score, and the top 3 recommendations are returned with their match percentage and required skills

## Dataset

The current version uses a small in-memory dataset of five sample job roles (Data Scientist, DevOps Engineer, Backend Developer, Machine Learning Engineer, Cloud Architect), each with a comma-separated list of required skills. Swapping in a larger dataset is as simple as replacing the in-memory `DataFrame` with `pd.read_csv()`.

## Project Structure

```text
DecodeLabs_Career_Recommender/
├── recommendation_logic.py   # Main script
└── README.md                 # Project documentation
```

## Getting Started

```bash
pip install pandas scikit-learn
python recommendation_logic.py
```

The script lists the available skills, prompts you to enter your own (comma-separated), then prints the top 3 recommended career paths with match scores and required skills.

## What I Learned

TF-IDF makes a big difference over plain keyword matching — it naturally down-weights common terms like "Python" that appear everywhere and up-weights rarer, more specific skills that actually distinguish one role from another. Adding the synonym normalization step also showed me how much real-world input varies from clean, well-formatted data — users rarely type things exactly the way a database expects.

## Author

Wireko Fosu Eric — AI Intern, DecodeLabs.
