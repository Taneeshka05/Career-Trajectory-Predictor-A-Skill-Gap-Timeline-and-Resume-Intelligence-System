from core import config as cfg

PREREQUISITES = {
    "pandas": ["python"],
    "numpy": ["python"],
    "scikit learn": ["python", "numpy"],
    "machine learning": ["python", "statistics"],
    "feature engineering": ["pandas", "machine learning"],
    "hypothesis testing": ["statistics"],
    "deep learning": ["machine learning"],
    "tensorflow": ["deep learning"],
    "pytorch": ["deep learning"],
    "model deployment": ["machine learning"],
    "mlops": ["docker", "model deployment"],
    "javascript": ["html"],
    "css": ["html"],
    "bootstrap": ["html", "css"],
    "react": ["javascript", "css"],
    "typescript": ["javascript"],
    "nodejs": ["javascript"],
    "spring boot": ["java"],
    "hibernate": ["java", "sql"],
    "junit": ["java"],
    "maven": ["java"],
    "microservices": ["spring boot", "rest api"],
    "mysql": ["sql"],
    "kubernetes": ["docker"],
    "terraform": ["aws"],
    "ansible": ["linux"],
    "bash": ["linux"],
    "monitoring": ["linux"],
    "jenkins": ["git"],
    "selenium": ["manual testing"],
    "automation testing": ["manual testing"],
    "api testing": ["rest api"],
}


def learning_order(missing, owned):
    owned = set(owned)
    needed = set()

    def collect(skill):
        if skill in owned or skill in needed:
            return
        needed.add(skill)
        for p in PREREQUISITES.get(skill, []):
            collect(p)

    for s in missing:
        collect(s)

    order = []
    seen = set()

    def visit(skill):
        if skill in seen:
            return
        seen.add(skill)
        for p in sorted(PREREQUISITES.get(skill, []), key=lambda x: (cfg.DIFFICULTY[x], x)):
            if p in needed:
                visit(p)
        order.append(skill)

    for s in sorted(needed, key=lambda x: (cfg.DIFFICULTY[x], x)):
        visit(s)
    return order


def dependency_dot(order, owned, missing):
    nodes = set(order)
    edges = []
    for s in order:
        for p in PREREQUISITES.get(s, []):
            nodes.add(p)
            edges.append((p, s))
    lines = ["digraph G {", "rankdir=LR;",
             'node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    for n in sorted(nodes):
        if n in owned:
            color = "#bbf7d0"
        elif n in missing:
            color = "#fecaca"
        else:
            color = "#fed7aa"
        lines.append(f'"{n}" [fillcolor="{color}"];')
    for a, b in edges:
        lines.append(f'"{a}" -> "{b}";')
    lines.append("}")
    return "\n".join(lines)