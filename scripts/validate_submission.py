import yaml, json, ast, os, sys

print("=" * 56)
print("  FINAL SUBMISSION VALIDATION")
print("=" * 56)

all_ok = True

def check(label, result):
    global all_ok
    icon = "pass" if result else "FAIL"
    if not result:
        all_ok = False
    print(f"  [{icon}] {label}")

# 1. openenv.yaml
print("\n[openenv.yaml]")
with open("openenv.yaml") as f:
    oe = yaml.safe_load(f)
env = oe["environment"]
check("name = tribunal-env", env.get("name") == "tribunal-env")
check("version = 1.0.0", env.get("version") == "1.0.0")
check("theme = multi-agent-interactions", env.get("theme") == "multi-agent-interactions")
check("port = 7860", env.get("port") == 7860)
check("entrypoint contains server:app", "server:app" in str(env.get("entrypoint", "")))
check("description > 50 chars", len(env.get("description", "")) > 50)
check("endpoints >= 5", len(env.get("endpoints", {})) >= 5)
check("metadata.links present", "links" in env.get("metadata", {}))

# 2. Dockerfile
print("\n[Dockerfile]")
df = open("Dockerfile").read()
check("Stage 1 - node:20-alpine", "node:20-alpine" in df)
check("Stage 2 - python:3.11-slim", "python:3.11-slim" in df)
check("npm run build", "npm run build" in df)
check("COPY --from=dashboard-builder", "--from=dashboard-builder" in df)
check("port 7860", "7860" in df)
check("uvicorn CMD", "uvicorn" in df)
check("COPY shared/", "COPY shared/" in df)
check("COPY client/", "COPY client/" in df)

# 3. README frontmatter
print("\n[README.md HF frontmatter]")
readme = open("README.md").read()
check("YAML frontmatter", readme.startswith("---"))
check("sdk: docker", "sdk: docker" in readme)
check("app_port: 7860", "app_port: 7860" in readme)
check("emoji present", "emoji:" in readme)
check("pinned: true", "pinned: true" in readme)
check("Submission checklist", "- [x]" in readme)
check("Mermaid diagram", "mermaid" in readme)
check("Anti-hack section", "Anti-Hack" in readme)
check("Results table", "Random" in readme)

# 4. Required files
print("\n[Required files]")
required = [
    "scripts/push_to_hf.sh",
    "assets/video_script.md",
    "assets/blog_post.md",
    "assets/before_after_examples.md",
    ".github/workflows/ci.yml",
    "scripts/eval_judge.py",
    "scripts/smoke_test.py",
    "docker-compose.yml",
    "notebooks/train_grpo.ipynb",
    "openenv.yaml",
]
for f in required:
    check(f, os.path.isfile(f))

# 5. CI workflow
print("\n[CI workflow]")
ci_text = open(".github/workflows/ci.yml").read()
check("ruff check", "ruff check" in ci_text)
check("black --check", "black --check" in ci_text)
check("pytest", "pytest" in ci_text)
check("smoke_test", "smoke_test" in ci_text)
check("eval_judge", "eval_judge" in ci_text)

# 6. Port consistency
print("\n[Port consistency - 7860]")
check("openenv.yaml", "7860" in open("openenv.yaml").read())
check("Dockerfile", "7860" in open("Dockerfile").read())
check("docker-compose.yml", "7860" in open("docker-compose.yml").read())
check("README.md", "7860" in open("README.md").read())
check("server.py reads TRIBUNAL_PORT", "TRIBUNAL_PORT" in open("src/tribunal/server.py").read())

# 7. Python syntax
print("\n[Python syntax]")
errors = 0
total = 0
for root, dirs, files in os.walk("."):
    if ".git" in root or "node_modules" in root or ".venv" in root:
        continue
    for f in files:
        if f.endswith(".py"):
            total += 1
            try:
                ast.parse(open(os.path.join(root, f)).read())
            except SyntaxError as e:
                errors += 1
                print(f"  [FAIL] {os.path.join(root, f)}: {e}")

check(f"{total} Python files, {errors} syntax errors", errors == 0)

print("\n" + "=" * 56)
if all_ok:
    print("  ALL CHECKS PASSED")
else:
    print("  SOME CHECKS FAILED")
print("=" * 56)

sys.exit(0 if all_ok else 1)
