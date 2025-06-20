import os
import json
import xml.etree.ElementTree as ET
import re
from collections import Counter
from itertools import combinations
from datetime import datetime
from nltk.corpus import stopwords
from nltk import download, ngrams

# === SETUP ===
try:
    stopwords.words("english")
except LookupError:
    print("Downloading NLTK stopwords...")
    download("stopwords")

DIRECTORY = "./xml_jobs"

OUTPUT_FILENAME = "job_analysis_results.json"

TOP_N_RESULTS = 100

CUSTOM_STOPWORDS = {
    'work', 'experience', 'team', 'job', 'position', 'role', 'company', 'organization',
    'looking', 'hiring', 'etc', 'new', 'join', 'skills', 'responsibilities',
    'requirements', 'qualifications', 'strong', 'ability', 'knowledge',
    'years', 'including', 'using', 'develop', 'design', 'business',
    'https', 'apply', 'weworkremotely.com', 'remote-jobs', 'url', 'http',
    'equal', 'sexual', 'orientation', 'gender', 'religion', 'color', 'opportunity',
    'race', 'sex', 'regard', 'consideration', 'national', 'origin', 'identity',
    'disability', 'veteran', 'status', 'employment', 'diversity', 'backgrounds'
}

KEY_SKILLS_FOR_COOCCURRENCE = [
    'python', 'java', 'javascript', 'c++', 'react', 'angular', 'vue',
    'node.js', 'sql', 'nosql', 'mongodb', 'postgres',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes',
    'tensorflow', 'pytorch', 'scikit-learn', 'pandas'
]

SENIORITY_LEVELS = {
    'Lead/Principal': ['lead', 'principal', 'staff', 'principal-engineer', 'lead-developer'],
    'Senior': ['senior', 'sr', 'sr.'],
    'Mid-Level': ['mid', 'mid-level', 'intermediate'],
    'Junior/Entry': ['junior', 'jr', 'jr.', 'entry', 'entry-level', 'graduate', 'intern']
}

SKILL_CATEGORIES = {
    'Programming Languages': ['python', 'java', 'javascript', 'typescript', 'go', 'golang', 'ruby', 'php', 'c++', 'c#', 'swift', 'kotlin', 'rust', 'scala'],
    'Frontend Frameworks': ['react', 'angular', 'vue', 'vue.js', 'svelte', 'next.js', 'remix'],
    'Backend Frameworks': ['node.js', 'django', 'flask', 'ruby-on-rails', 'spring', 'asp.net'],
    'Cloud Platforms': ['aws', 'azure', 'gcp', 'google-cloud', 'heroku', 'digitalocean', 'oracle-cloud'],
    'Databases': ['sql', 'mysql', 'postgresql', 'postgres', 'sqlite', 'mssql', 'oracle', 'nosql', 'mongodb', 'redis', 'cassandra', 'dynamodb'],
    'DevOps & CI/CD': ['docker', 'kubernetes', 'k8s', 'terraform', 'ansible', 'jenkins', 'gitlab-ci', 'github-actions', 'ci/cd'],
    'Data Science & ML': ['pandas', 'numpy', 'scipy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'ml', 'machine-learning', 'ai', 'jupyter'],
    'Big Data': ['spark', 'hadoop', 'kafka', 'flink', 'bigquery', 'redshift', 'snowflake'],
    'Testing': ['jest', 'mocha', 'cypress', 'selenium', 'phpunit', 'pytest'],
    'Mobile': ['ios', 'android', 'react-native', 'flutter']
}

SOFT_SKILLS = [
    'communication', 'collaborate', 'collaboration', 'teamwork', 'leadership', 'ownership',
    'mentorship', 'mentor', 'problem-solving', 'critical-thinking', 'agile', 'scrum',
    'proactive', 'self-starter', 'fast-paced', 'organized'
]

BENEFITS_AND_CULTURE = [
    'remote', 'hybrid', 'flexible', 'fully-remote', 'wfh', 'work-from-home',
    '401(k)', '401k', 'health', 'dental', 'vision', 'insurance', 'pto', 'unlimited-pto',
    'parental-leave', 'equity', 'stock-options', 'bonus', 'wellness'
]

STOPWORDS = set(stopwords.words("english"))
STOPWORDS.update(CUSTOM_STOPWORDS)


# === CLEANING & PROCESSING UTILS ===
def extract_and_clean(parent, element_name):
    text = parent.findtext(element_name, default="")
    clean_text = re.sub(r'<[^>]+>', '', text)
    clean_text = re.sub(r'&[a-z]+;', '', clean_text)
    return clean_text.lower()

def tokenize(text):
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9.+\-#()]*\b', text)
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]

def process_directory(directory):
    all_job_items = []
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return []
    
    files_to_process = [f for f in os.listdir(directory) if f.endswith((".rss", ".xml"))]
    if not files_to_process:
        print(f"Warning: No .rss or .xml files found in '{directory}'.")
        return []

    print("\nProcessing files...")
    for filename in files_to_process:
        full_path = os.path.join(directory, filename)
        try:
            tree = ET.parse(full_path)
            root = tree.getroot()
            for item in root.findall(".//item"):
                job_data = {
                    'title_text': extract_and_clean(item, 'title'),
                    'description_text': extract_and_clean(item, 'description')
                }
                job_data['description_tokens'] = tokenize(job_data['description_text'])
                all_job_items.append(job_data)
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
    return all_job_items

def analyze_job_titles(all_job_items, top_n):
    all_titles = [item['title_text'] for item in all_job_items]
    title_tokens = [token for title in all_titles for token in title.lower().split()]
    
    bigrams = Counter(ngrams(title_tokens, 2))
    trigrams = Counter(ngrams(title_tokens, 3))
    
    top_bigrams = [{"role": " ".join(b), "frequency": f} for b, f in bigrams.most_common(top_n)]
    top_trigrams = [{"role": " ".join(t), "frequency": f} for t, f in trigrams.most_common(top_n)]
    
    return {'top_bigrams': top_bigrams, 'top_trigrams': top_trigrams}

def analyze_by_keyword_list(all_tokens, keyword_list):
    keyword_set = set(keyword_list)
    counter = Counter(token for token in all_tokens if token in keyword_set)
    return [{"term": k, "frequency": f} for k, f in counter.most_common()]

def analyze_seniority(all_tokens, seniority_map):
    seniority_counts = Counter()
    for token in all_tokens:
        for level, keywords in seniority_map.items():
            if token in keywords:
                seniority_counts[level] += 1
    return [{"level": l, "frequency": f} for l, f in seniority_counts.most_common()]

def analyze_categorized_skills(all_job_items, category_map):
    category_counts = Counter()
    skill_to_category = {skill: category for category, skills in category_map.items() for skill in skills}
    
    for item in all_job_items:
        mentioned_categories = {skill_to_category[token] for token in item['description_tokens'] if token in skill_to_category}
        category_counts.update(mentioned_categories)
        
    return [{"category": c, "jobs_mentioned_in": f} for c, f in category_counts.most_common()]

# === MAIN SCRIPT ===
if __name__ == "__main__":
    all_jobs = process_directory(DIRECTORY)
    
    if not all_jobs:
        print("\nNo job data processed. Exiting.")
    else:
        print(f"\nSuccessfully processed {len(all_jobs)} job descriptions.")
        
        # --- PREPARE DATA FOR ANALYSIS ---
        all_description_tokens = [token for item in all_jobs for token in item['description_tokens']]
        
        # --- RUN ALL ANALYSES ---
        print("Running analyses...")
        word_freq = Counter(all_description_tokens)
        bigram_freq = Counter(ngrams(all_description_tokens, 2))
        trigram_freq = Counter(ngrams(all_description_tokens, 3))

        cooccurrence_counter = Counter()
        key_skills_set = set(KEY_SKILLS_FOR_COOCCURRENCE)
        for item in all_jobs:
            present_skills = key_skills_set.intersection(item['description_tokens'])
            if len(present_skills) >= 2:
                for pair in combinations(sorted(list(present_skills)), 2):
                    cooccurrence_counter[pair] += 1

        job_title_results = analyze_job_titles(all_jobs, TOP_N_RESULTS)
        seniority_results = analyze_seniority(all_description_tokens, SENIORITY_LEVELS)
        categorized_skills_results = analyze_categorized_skills(all_jobs, SKILL_CATEGORIES)
        soft_skills_results = analyze_by_keyword_list(all_description_tokens, SOFT_SKILLS)
        benefits_results = analyze_by_keyword_list(all_description_tokens, BENEFITS_AND_CULTURE)

        # --- STRUCTURE DATA FOR JSON EXPORT ---
        print("Structuring data for export...")
        results_data = {
            "metadata": {
                "total_jobs_processed": len(all_jobs),
                "analysis_timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "top_n_results_per_category": TOP_N_RESULTS
            },
            "core_analyses": {
                "top_unigrams": [{"term": w, "frequency": f} for w, f in word_freq.most_common(TOP_N_RESULTS)],
                "top_bigrams": [{"term": " ".join(b), "frequency": f} for b, f in bigram_freq.most_common(TOP_N_RESULTS)],
                "top_trigrams": [{"term": " ".join(t), "frequency": f} for t, f in trigram_freq.most_common(TOP_N_RESULTS)],
                "top_skill_cooccurrences": [{"pair": " & ".join(p), "frequency": f} for p, f in cooccurrence_counter.most_common(TOP_N_RESULTS)],
            },
            "employer_insight_analyses": {
                "job_title_analysis": job_title_results,
                "seniority_level_distribution": seniority_results,
                "most_in_demand_skill_categories": categorized_skills_results,
                "most_requested_soft_skills": soft_skills_results,
                "most_mentioned_benefits_and_culture": benefits_results,
            }
        }
        
        # --- EXPORT TO JSON FILE ---
        try:
            with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=4)
            print(f"\nAll analyses successfully exported to {OUTPUT_FILENAME}")
        except Exception as e:
            print(f"\nError exporting to JSON: {e}")