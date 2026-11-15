# Career Trajectory Predictor & Skill Gap Timeline Analyzer

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit_1.51-FF4B4B.svg)](https://streamlit.io/)
[![ML: scikit-learn](https://img.shields.io/badge/Machine%20Learning-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![Deep Learning: Keras/TF](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-orange.svg)](https://www.tensorflow.org/)
[![Tests: 34 Passed](https://img.shields.io/badge/Tests-34%20Passed-brightgreen.svg)](https://pytest.org/)

An intelligent, data-driven career intelligence system that bridges the gap between candidate resumes and modern industry job requirements. Built for academic excellence, technical explainability, and practical career navigation.

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Core Capabilities](#core-capabilities)
4. [Real Machine Learning Pipeline (Phase 4 & 5)](#real-machine-learning-pipeline)
5. [Skill Taxonomy & Transferability Matrix](#skill-taxonomy--transferability-matrix)
6. [NLP Resume Quality Engine](#nlp-resume-quality-engine)
7. [Mathematical Formulations](#mathematical-formulations)
8. [Dataset Intelligence & Provenance](#dataset-intelligence--provenance)
9. [College Viva Defense Q&A Guide](#college-viva-defense-qa-guide)
10. [Local Setup & Execution Instructions](#local-setup--execution-instructions)
11. [Testing & Quality Assurance Report](#testing--quality-assurance-report)

---

## 1. Project Overview

The **Career Trajectory Predictor & Skill Gap Timeline Analyzer** is an end-to-end career analytics platform. Rather than providing simple keyword matching or arbitrary similarity percentages, the platform implements:

- **Multi-Dimensional Compatibility Scoring**: Direct technical skill overlap (40%), transferable skill credit (15%), TF-IDF semantic cosine similarity (20%), soft skills alignment (15%), and experience level fit (10%).
- **Cross-Domain Transferable Skill Bridges**: A pedagogical transferability matrix that credits related competencies (e.g., PyTorch provides an 85% transferable foundation for TensorFlow; Power BI provides an 85% foundation for Tableau).
- **Prerequisite-Aware Dynamic Timeline Forecasting**: A Directed Acyclic Graph (DAG) topological sort that schedules foundation skills before advanced topics, dynamically adjusting timelines based on weekly study availability (5 to 40 hours/week).
- **Career Trajectory Progression**: Simulates milestone-by-milestone growth from *Junior* to *Mid-Level* to *Senior* readiness tiers using trained classification models.
- **Genuine Machine Learning Pipeline**: A leakage-free data validation, EDA, preprocessing, model training, comparison, and live inference engine supporting both user-uploaded datasets and bundled benchmarks.
- **Action-Oriented NLP Resume Improver**: Detects 20+ passive constructions, quantifies achievements, evaluates vocabulary richness (Type-Token Ratio), and generates authentic STAR/CAR bullet frameworks.
- **Privacy-Preserving SQLite History**: Optional opt-in local persistence for tracking roadmap milestones.

---

## 2. High-Level Architecture

```
                                +-----------------------------+
                                |      Candidate Inputs       |
                                | (.txt, .pdf, .docx, pasted) |
                                +--------------+--------------+
                                               |
                                               v
                                +-----------------------------+
                                | Multi-Format Document Parser|
                                |  (core/parser.py - Regex/ET)|
                                +--------------+--------------+
                                               |
                +------------------------------+-------------------------------+
                |                                                              |
                v                                                              v
+-------------------------------+                            +-------------------------------+
|     Text & NLP Preprocessing  |                            |     Skill Taxonomy Engine     |
|   (Stopwords, Cleaning, TFIDF)|                            | (Aliases, 6 Categories, Trans)|
+---------------+---------------+                            +---------------+---------------+
                |                                                              |
                +------------------------------+-------------------------------+
                                               |
                                               v
                                +-----------------------------+
                                |    Career Analytics Engine  |
                                |      (core/engine.py)       |
                                +--------------+--------------+
                                               |
         +-------------------------------------+------------------------------------+
         |                                     |                                    |
         v                                     v                                    v
+------------------+                 +--------------------+               +--------------------+
| ML Regressors    |                 | ML Classifiers     |               | Keras Neural Net   |
| (Linear, Ridge)  |                 | (Logistic, RF, KNN)|               | (Deep Language NN) |
| &rarr; Weeks Est.|                 | &rarr; Readiness   |               | &rarr; Style Score |
+--------+---------+                 +---------+----------+               +---------+----------+
         |                                     |                                    |
         +-------------------------------------+------------------------------------+
                                               |
                                               v
                                +-----------------------------+
                                |  Explainable Output & Visual|
                                | Gauges, DAG, Radar, Gantt   |
                                +-----------------------------+
```

### Module Breakdown
- `core/engine.py`: Central coordination engine for multi-dimensional compatibility, career screening, and trajectory evaluation.
- `core/taxonomy.py`: Structured skill taxonomy across 6 technical categories, 50+ aliases, soft skills, and cross-domain transferability graph.
- `core/parser.py`: Multi-format resume parser supporting `.txt`, `.pdf` (via `pypdf`), and native pure-Python `.docx` extraction (via `zipfile` + `xml.etree.ElementTree`); validates uploaded resume content and derives experience from resume text and employment dates.
- `core/ml_pipeline.py`: Real-time machine learning pipeline: DataValidator, EDA, SimpleImputer, StandardScaler, OneHotEncoder, regression/classification training, unseen test metrics, and live inference.
- `core/nlp_feedback.py`: Passive verb replacement, vocabulary analysis (TTR), quantifiable metric detection, and STAR bullet point generation.
- `core/prerequisites.py`: DAG prerequisite dependencies and topological sorting for ordered learning roadmaps.
- `core/trajectory.py`: What-If simulator, trajectory path evaluation, and skill acquisition ROI calculator.
- `core/models.py`: Pre-trained model artifact management, 5-fold cross-validation, learning curves, and noise stress tests.
- `core/charts.py`: Plotly charts (Gauges, Radar, Gantt roadmaps, Trajectory projections, Confusion Matrices, EDA heatmaps).
- `core/database.py`: SQLite persistence for analysis history and milestone tracking (requires `CAREERPATH_ENABLE_HISTORY=1`).
- `core/ui.py`: Unified dark professional theme, glassmorphic cards, badges, and responsive UI components.

---

## 3. Core Capabilities

### 1. Intelligent Job Compatibility Matching
Evaluates compatibility across multiple explainable pillars:
- **Direct Skill Overlap**: Ratio of mandatory skills present in the resume.
- **Transferable Skill Credit**: Credits related skills using our transferability matrix (e.g. PyTorch for TensorFlow).
- **Textual Semantic Similarity**: Cosine similarity of unigram + bigram TF-IDF vectors.
- **Soft Skills Alignment**: Matches collaboration, communication, agile methodologies, and leadership.
- **Experience Fit**: Compares candidate years against role seniority benchmarks (Junior: 0-2 yrs, Mid: 3-5 yrs, Senior: 6+ yrs).

### 2. Advanced Skill Gap & Taxonomy
- 52 core technical skills categorized into:
  - *Programming Languages* (Python, Java, JavaScript, TypeScript, Bash, SQL)
  - *Data & Analytics* (Excel, Tableau, Power BI, Statistics, Pandas, NumPy, Data Visualization, Data Cleaning, Reporting)
  - *Machine Learning & AI* (Scikit-Learn, Machine Learning, Deep Learning, TensorFlow, PyTorch, Feature Engineering, MLOps, Model Deployment)
  - *Web & Full-Stack* (HTML, CSS, Bootstrap, React, Node.js, REST API, MongoDB, MySQL, Spring Boot, Hibernate, Microservices)
  - *Cloud & DevOps* (Linux, Docker, Kubernetes, Jenkins, AWS, Terraform, Ansible, Monitoring)
  - *Software Testing & QA* (Manual Testing, Selenium, JUnit, Jira, Test Cases, API Testing, Postman, Automation Testing)
- Distinguishes **Core Requirements** from **Secondary Skills**.
- Explains the exact role context and estimated learning hours for every gap.

### 3. Cross-Domain Transferable Skill Bridges
Prevents penalizing candidates who know alternative technologies.
| Missing Target Skill | Candidate Related Skill | Transferability Credit | Pedagogical Rationale |
| :--- | :--- | :--- | :--- |
| **TensorFlow** | PyTorch | 85% | Shared tensor computations, backpropagation, and autograd architectures. |
| **Tableau** | Power BI | 85% | Shared data modeling, DAX/LOD expressions, and dashboard storytelling. |
| **MySQL** | SQL (PostgreSQL/Oracle) | 95% | Direct relational algebra, ANSI SQL compliance, and indexing strategies. |
| **Docker** | Kubernetes | 80% | Container lifecycle fundamentals and microservice isolation. |
| **Automation Testing**| Selenium | 85% | Test automation frameworks, element locators, and assertion suites. |

### 4. Dynamic Timeline Predictor & Gantt Roadmap
- Uses topological sorting on prerequisite dependencies:
  $$\text{Python} \rightarrow \text{Statistics} \rightarrow \text{Machine Learning} \rightarrow \text{Deep Learning} \rightarrow \text{TensorFlow}$$
- Automatically adds prerequisite foundation skills in amber if missing.
- Dynamically scales based on learner commitment:
  $$\text{Total Duration (Weeks)} = \frac{\sum \text{Base Weeks}}{\text{Weekly Hours}} \times 10$$

### 5. Career What-If Simulator & Trajectory Lab
- Interactively select hypothetical skills to acquire (e.g. *"What happens if I learn Docker and AWS?"*).
- Immediately re-evaluates the trained ML models to compute:
  - Skill match gain
  - Text similarity gain
  - Readiness tier promotion (e.g. Junior &rarr; Mid-Level)
  - Net weeks saved

---

## 4. Real Machine Learning Pipeline

The project includes an interactive, production-grade Machine Learning engine (`pages/5_Model_Lab.py` & `core/ml_pipeline.py`) implementing the full lifecycle:

```
DATA INPUT (CSV/Excel)
    &darr;
DATA VALIDATION (Health score, types, duplicates, null rates)
    &darr;
EXPLORATORY DATA ANALYSIS (Summary statistics, correlation matrix, missing bar charts)
    &darr;
PREPROCESSING (SimpleImputer, StandardScaler, OneHotEncoder - fit on train only)
    &darr;
TRAIN / TEST SPLIT (Stratified splitting to prevent data leakage)
    &darr;
MODEL TRAINING & COMPARISON
  - Regression: Linear Regression, Ridge (L2), Random Forest Regressor
  - Classification: Logistic Regression, Random Forest Classifier, K-Nearest Neighbors
    &darr;
UNSEEN TEST EVALUATION (R², RMSE, MAE, Accuracy, F1-Score, Confusion Matrix)
    &darr;
FEATURE IMPORTANCE & LIVE PREDICTION SANDBOX
```

Users can test this pipeline using their own uploaded datasets or the bundled `career_ml_sample_dataset.csv` (650 records).

---

## 5. NLP Resume Quality Engine

- **Active Verb Engineering**: Identifies passive/weak constructions (`responsible for`, `worked on`, `duties included`, `assisted in`, `tasked with`, `handled`) and replaces them with impact verbs (`led`, `developed`, `orchestrated`, `engineered`, `spearheaded`).
- **Vocabulary Diversity (Type-Token Ratio)**:
  $$\text{TTR} = \frac{\text{Unique Lexical Tokens}}{\text{Total Tokens}}$$
- **Quantifiable Metrics Detection**: Recognizes percentages, metrics, monetary savings, user scale, and latency reductions (e.g. `35%`, `5,000+ users`, `10k requests`).
- **STAR Bullet Point Generator**: Generates Situation-Task-Action-Result frameworks mapped to candidate skills.

---

## 6. Mathematical Formulations

### 1. TF-IDF Representation
For term $t$ in document $d$ within corpus $D$:
$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### 2. Cosine Similarity
$$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$

### 3. Ridge Regression Cost Function (L2 Regularization)
Prevents multicollinearity and overfitting in time-to-ready estimation:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2} \sum_{j=1}^n \theta_j^2$$

### 4. Binary Cross-Entropy (Language Quality Neural Network)
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]$$

---

## 7. Dataset Intelligence & Provenance

1. **`resumes_dataset.csv`**: 5,500 profiles spanning 7 engineering domains, containing cleaned text, extracted skills, experience years, and active language labels.
2. **`jobs_dataset.csv`**: 420 job postings (60 per domain) representing real-world role requirements.
3. **`skill_reference.csv`**: 52 canonical skills with difficulty weights (1 to 5) and base learning weeks.
4. **`career_ml_sample_dataset.csv`**: 650 records curated for dynamic regression (`preparation_weeks`) and classification (`readiness_tier`) benchmarking.

*Honest Academic Disclosure:* Synthetic resumes and job postings are generated using controlled templates to provide deterministic benchmark evaluation.

---

## 8. College Viva Defense Q&A Guide

**Q1: Why not just use Cosine Similarity for resume matching?**
*Answer:* Cosine similarity measures textual overlap but ignores skill criticality, prerequisites, and transferability. A resume might share vocabulary with a job description while lacking the core mandatory programming language. Our system decouples semantic text similarity from technical skill compatibility and transferable competency bridges.

**Q2: How do you prevent data leakage during model training?**
*Answer:* Preprocessing transformers (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`) are fitted strictly on the training partition after performing `train_test_split`. The test set is transformed using the training parameters, ensuring unseen test metrics reflect genuine generalization.

**Q3: How is the learning roadmap ordered?**
*Answer:* Skill prerequisites form a Directed Acyclic Graph (DAG). We apply topological sorting (Kahn's algorithm / DFS) so that prerequisite foundations (e.g. Python & Statistics) are scheduled before advanced applied topics (e.g. Machine Learning & TensorFlow).

**Q4: How does the What-If Simulator work?**
*Answer:* The simulator adds hypothetical skills to the candidate's feature vector, runs prerequisite resolution to identify required foundation skills, updates feature values (e.g. `missing_n`, `skill_match_ratio`), and re-executes the trained Ridge Regressor and Logistic Classifier to produce updated predictions.

---

## 9. Local Setup & Execution Instructions

### Prerequisites
- Python 3.11 (Recommended)
- PowerShell (Windows) or Bash (macOS/Linux)

### 1. Clone & Set Up Virtual Environment
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Launch the Application
```bash
# Standard Launch:
python -m streamlit run app.py

# Optional: Enable SQLite History Storage:
$env:CAREERPATH_ENABLE_HISTORY="1"
python -m streamlit run app.py
```

### 4. Run Test Suite
```bash
python -m pytest tests/ -v
```

---

## 10. Testing & Quality Assurance Report

The project includes an automated test suite executed via `pytest`. All 34 tests pass with zero regressions:

| Test Module | Coverage Scope | Status |
| :--- | :--- | :--- |
| `tests/test_database_privacy.py` | SQLite opt-in guard, environment variable privacy, rejection of disabled storage | **4 Passed** |
| `tests/test_engine.py` | Analysis score bounds, DAG prerequisite ordering, timeline scaling, trajectory monotonicity, feature explainability | **10 Passed** |
| `tests/test_ml_pipeline.py` | DataValidator audit, empty file rejection, dynamic regression pipeline, dynamic classification pipeline, live inference | **4 Passed** |
| `tests/test_nlp_enhanced.py` | Expanded weak verb replacement, vocabulary richness (TTR), STAR bullet generator, rubric scoring | **4 Passed** |
| `tests/test_parser.py` | Native pure-Python DOCX extraction, multi-format dispatcher, metadata extraction (degrees, certs, years) | **3 Passed** |
| `tests/test_taxonomy.py` | Taxonomy categories, alias mapping ('js' &rarr; 'javascript', 'postgres' &rarr; 'sql'), soft skills, transferable credits | **6 Passed** |
| `tests/test_career_intelligence.py` | Multi-dimensional compatibility, transferable bridge integration, recruiter screening | **3 Passed** |
| **Total** | **Comprehensive Regression & Feature Testing** | **34 Passed (100%)** |
