from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = BASE_DIR / "careerpath.db"
for folder in (DATA_DIR, ARTIFACT_DIR, LOG_DIR):
    folder.mkdir(exist_ok=True)

SEED = 42
N_RESUMES = 5500
JOBS_PER_DOMAIN = 60
HOURS_BASELINE = 10

DOMAIN_SKILLS = {
    "Data Analyst": ["excel", "sql", "python", "tableau", "power bi", "statistics", "pandas",
                     "data visualization", "data cleaning", "reporting"],
    "Data Scientist": ["python", "statistics", "machine learning", "pandas", "numpy", "scikit learn",
                       "sql", "data visualization", "feature engineering", "hypothesis testing"],
    "ML Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch",
                    "scikit learn", "numpy", "docker", "mlops", "model deployment"],
    "Web Developer": ["html", "css", "javascript", "react", "nodejs", "git", "rest api",
                      "mongodb", "bootstrap", "typescript"],
    "Java Developer": ["java", "spring boot", "hibernate", "sql", "maven", "rest api", "junit",
                       "git", "microservices", "mysql"],
    "DevOps Engineer": ["linux", "docker", "kubernetes", "jenkins", "aws", "terraform", "git",
                        "ansible", "bash", "monitoring"],
    "Software Tester": ["manual testing", "selenium", "junit", "jira", "test cases", "api testing",
                        "sql", "python", "postman", "automation testing"],
}
DOMAIN_NAMES = list(DOMAIN_SKILLS.keys())
ALL_SKILLS = sorted({s for v in DOMAIN_SKILLS.values() for s in v})

DIFFICULTY = {
    "excel": 1, "sql": 2, "python": 3, "tableau": 2, "power bi": 2, "statistics": 3, "pandas": 2,
    "data visualization": 2, "data cleaning": 2, "reporting": 1, "machine learning": 4, "numpy": 2,
    "scikit learn": 3, "feature engineering": 4, "hypothesis testing": 3, "deep learning": 5,
    "tensorflow": 4, "pytorch": 4, "docker": 3, "mlops": 5, "model deployment": 4, "html": 1,
    "css": 1, "javascript": 3, "react": 3, "nodejs": 3, "git": 1, "rest api": 3, "mongodb": 2,
    "bootstrap": 1, "typescript": 3, "java": 4, "spring boot": 4, "hibernate": 4, "maven": 2,
    "junit": 2, "microservices": 5, "mysql": 2, "linux": 2, "kubernetes": 5, "jenkins": 3,
    "aws": 4, "terraform": 4, "ansible": 3, "bash": 2, "monitoring": 3, "manual testing": 1,
    "selenium": 3, "jira": 1, "test cases": 1, "api testing": 3, "postman": 1,
    "automation testing": 4,
}
SKILL_WEEKS = {s: DIFFICULTY[s] * 2 + 1 for s in ALL_SKILLS}

STRONG_VERBS = ["developed", "designed", "improved", "automated", "reduced", "delivered",
                "optimized", "launched", "implemented", "streamlined", "created", "engineered"]
WEAK_PHRASES = ["responsible for", "worked on", "helped with", "involved in", "duties included",
                "assisted in"]
REPLACEMENTS = {
    "responsible for": "led",
    "worked on": "developed",
    "helped with": "delivered",
    "involved in": "implemented",
    "duties included": "delivered",
    "assisted in": "improved",
}
THINGS = ["dashboards", "data pipelines", "web pages", "backend services", "test suites",
          "deployment scripts", "prediction models", "client reports", "internal tools",
          "database queries"]
EDUCATION_LIST = ["BCA graduate", "MCA student", "BTech graduate", "MSc graduate", "BSc graduate"]
JOB_LEVELS = ["Junior", "Mid Level", "Senior"]

REG_FEATURES = ["missing_n", "missing_difficulty", "years_exp", "match_score",
                "skill_match_ratio", "n_resume_skills"]
CLF_FEATURES = ["match_score", "skill_match_ratio", "matched_n", "missing_n", "years_exp",
                "n_resume_skills", "avg_diff_matched"]

DEMO_RESUME = ("BCA graduate with 1 years of experience. Technical skills include python, sql, "
               "excel. Created dashboards using excel. Responsible for data cleaning using "
               "python. Worked on database queries using sql.")
DEMO_YEARS = 1