"""
taxonomy.py
Comprehensive Skill Taxonomy, Categories, Aliases, and Transferability Matrix
for the Career Trajectory Predictor & Skill Gap Timeline Analyzer.
"""

from typing import Dict, List, Set, Tuple
from core import config as cfg

# 1. Categorized Skill Taxonomy
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "bash", "sql"
    ],
    "Data & Analytics": [
        "excel", "sql", "tableau", "power bi", "statistics", "pandas",
        "numpy", "data visualization", "data cleaning", "reporting"
    ],
    "Machine Learning & AI": [
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "scikit learn", "feature engineering", "hypothesis testing",
        "mlops", "model deployment"
    ],
    "Web & Full-Stack Development": [
        "html", "css", "javascript", "typescript", "react", "nodejs",
        "git", "rest api", "mongodb", "bootstrap", "spring boot",
        "hibernate", "microservices", "mysql"
    ],
    "Cloud & DevOps": [
        "linux", "docker", "kubernetes", "jenkins", "aws", "terraform",
        "ansible", "bash", "monitoring", "git"
    ],
    "Software Testing & QA": [
        "manual testing", "selenium", "junit", "jira", "test cases",
        "api testing", "postman", "automation testing"
    ],
}

# 2. Soft Skills & Professional Competencies
SOFT_SKILLS = [
    "communication", "leadership", "problem solving", "teamwork",
    "critical thinking", "agile", "scrum", "time management",
    "collaboration", "analytical thinking", "adaptability", "mentoring"
]

# 3. Aliases and Synonyms Dictionary
# Maps common abbreviations, alternative spellings, and tool variations to canonical skills
SKILL_ALIASES: Dict[str, str] = {
    # Languages
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "ts": "typescript",
    "sh": "bash",
    "shell": "bash",
    "shell scripting": "bash",
    "bash scripting": "bash",
    "sql query": "sql",
    "sql queries": "sql",
    "structured query language": "sql",
    "postgres": "sql",
    "postgresql": "sql",
    "oracle": "sql",
    "sqlite": "sql",
    
    # Data & Analytics
    "ms excel": "excel",
    "microsoft excel": "excel",
    "spreadsheets": "excel",
    "vlookup": "excel",
    "powerbi": "power bi",
    "ms power bi": "power bi",
    "tableau desktop": "tableau",
    "data viz": "data visualization",
    "dataviz": "data visualization",
    "eda": "data cleaning",
    "data wrangling": "data cleaning",
    "data preprocessing": "data cleaning",
    "bi reporting": "reporting",
    "executive reporting": "reporting",
    "statistical analysis": "statistics",
    "probability": "statistics",
    "inferential statistics": "statistics",
    "descriptive statistics": "statistics",
    
    # ML & AI
    "ml": "machine learning",
    "dl": "deep learning",
    "neural networks": "deep learning",
    "ann": "deep learning",
    "cnn": "deep learning",
    "rnn": "deep learning",
    "transformer": "deep learning",
    "transformers": "deep learning",
    "tf": "tensorflow",
    "torch": "pytorch",
    "sklearn": "scikit learn",
    "scikit-learn": "scikit learn",
    "feature selection": "feature engineering",
    "ab testing": "hypothesis testing",
    "a/b testing": "hypothesis testing",
    "hypothesis test": "hypothesis testing",
    "model serving": "model deployment",
    "deployment": "model deployment",
    "ml ops": "mlops",
    
    # Web & Full Stack
    "html5": "html",
    "css3": "css",
    "tailwind": "css",
    "reactjs": "react",
    "react.js": "react",
    "node": "nodejs",
    "node.js": "nodejs",
    "express": "nodejs",
    "expressjs": "nodejs",
    "mongo": "mongodb",
    "rest": "rest api",
    "restful": "rest api",
    "restful api": "rest api",
    "apis": "rest api",
    "microservice": "microservices",
    "micro service": "microservices",
    "spring": "spring boot",
    "springboot": "spring boot",
    "jpa": "hibernate",
    "github": "git",
    "version control": "git",
    
    # Cloud & DevOps
    "k8s": "kubernetes",
    "containerization": "docker",
    "containers": "docker",
    "amazon web services": "aws",
    "ec2": "aws",
    "s3": "aws",
    "iac": "terraform",
    "infrastructure as code": "terraform",
    "ci cd": "jenkins",
    "ci/cd": "jenkins",
    "continuous integration": "jenkins",
    "observability": "monitoring",
    "grafana": "monitoring",
    "prometheus": "monitoring",
    
    # Testing & QA
    "qa": "manual testing",
    "quality assurance": "manual testing",
    "test planning": "test cases",
    "test scripts": "test cases",
    "postman api": "postman",
    "rest assured": "api testing",
    "test automation": "automation testing",
    "automated testing": "automation testing",
    "unit testing": "junit",
    "unit tests": "junit",
    "agile scrum": "agile",
}

# 4. Transferable & Complementary Skills Matrix
# Quantifies cross-skill transferability (0.0 to 1.0)
TRANSFERABLE_SKILLS: Dict[Tuple[str, str], float] = {
    ("pytorch", "tensorflow"): 0.85,
    ("tensorflow", "pytorch"): 0.85,
    ("power bi", "tableau"): 0.85,
    ("tableau", "power bi"): 0.85,
    ("mysql", "sql"): 0.95,
    ("sql", "mysql"): 0.90,
    ("mongodb", "sql"): 0.60,
    ("sql", "mongodb"): 0.60,
    ("pandas", "numpy"): 0.80,
    ("numpy", "pandas"): 0.80,
    ("scikit learn", "machine learning"): 0.90,
    ("machine learning", "scikit learn"): 0.85,
    ("deep learning", "machine learning"): 0.85,
    ("machine learning", "deep learning"): 0.70,
    ("spring boot", "microservices"): 0.80,
    ("microservices", "spring boot"): 0.75,
    ("kubernetes", "docker"): 0.80,
    ("docker", "kubernetes"): 0.70,
    ("terraform", "ansible"): 0.70,
    ("ansible", "terraform"): 0.70,
    ("selenium", "automation testing"): 0.85,
    ("automation testing", "selenium"): 0.85,
    ("api testing", "postman"): 0.90,
    ("postman", "api testing"): 0.90,
    ("junit", "test cases"): 0.80,
    ("test cases", "junit"): 0.75,
    ("manual testing", "test cases"): 0.85,
    ("test cases", "manual testing"): 0.85,
    ("react", "javascript"): 0.85,
    ("javascript", "typescript"): 0.80,
    ("typescript", "javascript"): 0.90,
    ("bash", "linux"): 0.85,
    ("linux", "bash"): 0.80,
    ("jenkins", "docker"): 0.65,
    ("aws", "docker"): 0.65,
}


def get_skill_category(skill: str) -> str:
    """Returns the primary taxonomy category for a canonical skill."""
    for category, skills in SKILL_TAXONOMY.items():
        if skill in skills:
            return category
    return "Specialized Technology"


def find_transferable_credits(owned_skills: Set[str], missing_skills: Set[str]) -> List[Dict]:
    """
    Calculates transferable skill credits for missing target skills based on candidate skills.
    Returns list of dicts with target_skill, source_skill, credit_ratio, explanation.
    """
    credits = []
    for miss in missing_skills:
        best_credit = 0.0
        best_source = None
        for own in owned_skills:
            pair = (own, miss)
            if pair in TRANSFERABLE_SKILLS:
                rate = TRANSFERABLE_SKILLS[pair]
                if rate > best_credit:
                    best_credit = rate
                    best_source = own
        if best_credit > 0.0:
            credits.append({
                "target_skill": miss,
                "source_skill": best_source,
                "credit": best_credit,
                "percentage": int(best_credit * 100),
                "reason": (
                    f"Proficiency in '{best_source}' provides ~{int(best_credit * 100)}% "
                    f"transferable foundation for '{miss}'."
                )
            })
    return sorted(credits, key=lambda x: -x["credit"])


def categorize_skills(skills: List[str]) -> Dict[str, List[str]]:
    """Groups a list of skills by taxonomy category."""
    categorized = {cat: [] for cat in SKILL_TAXONOMY}
    categorized["Other Skills"] = []
    for s in skills:
        placed = False
        for cat, cat_skills in SKILL_TAXONOMY.items():
            if s in cat_skills:
                categorized[cat].append(s)
                placed = True
                break
        if not placed:
            categorized["Other Skills"].append(s)
    return {k: v for k, v in categorized.items() if v}
