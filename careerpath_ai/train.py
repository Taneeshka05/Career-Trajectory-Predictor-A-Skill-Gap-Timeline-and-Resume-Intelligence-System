import argparse
from core.models import train_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CareerPath AI models")
    parser.add_argument("--resumes", help="Optional CSV of real resumes (text column: Resume_str, Resume, resume_text or text)")
    parser.add_argument("--only-real", action="store_true", help="Use only the real resumes, no synthetic ones")
    args = parser.parse_args()
    train_all(args.resumes, args.only_real)
    print("Training finished. Artifacts saved in the artifacts folder.")