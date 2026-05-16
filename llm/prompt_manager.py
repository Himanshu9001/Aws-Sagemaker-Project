# Prompt Manager — version-controlled prompt templates
# Local files are source of truth
# Langfuse is the deployment registry
# Workflow: edit file → register to Langfuse → deploy to production label

import os
import json
from langfuse import Langfuse

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-d298883b-9e0a-46f3-b519-d7b1f1297af1"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-8238ff79-2ba7-46f7-97d8-f6e11cdc46df"
os.environ["LANGFUSE_HOST"]       = "https://cloud.langfuse.com"

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

langfuse = Langfuse(
    public_key="pk-lf-d298883b-9e0a-46f3-b519-d7b1f1297af1",
    secret_key="sk-lf-8238ff79-2ba7-46f7-97d8-f6e11cdc46df",
    host="https://cloud.langfuse.com"
)


def load_registry():
    with open(os.path.join(PROMPTS_DIR, "registry.json")) as f:
        return json.load(f)


def load_prompt_file(version, prompt_type):
    path = os.path.join(PROMPTS_DIR, version, f"{prompt_type}.txt")
    with open(path) as f:
        return f.read().strip()


def register_version(version):
    """Register all prompts for a version to Langfuse."""
    registry = load_registry()
    version_config = registry.get(version, {})

    for prompt_type, config in version_config.items():
        if prompt_type == "langfuse_name":
            continue
        template = load_prompt_file(version, prompt_type)
        try:
            prompt = langfuse.create_prompt(
                name=config["langfuse_name"],
                prompt=template,
                labels=[config["label"]],
                config={"description": config["description"], "version": version}
            )
            print(f"Registered {version}/{prompt_type}: {config['langfuse_name']} (v{prompt.version})")
        except Exception as e:
            print(f"Already exists or error — {config['langfuse_name']}: {e}")


def get_active_prompt(prompt_type, version=None):
    """Get compiled prompt template. Uses active_version if not specified."""
    registry   = load_registry()
    version    = version or registry["active_version"]
    config     = registry[version][prompt_type]

    try:
        prompt = langfuse.get_prompt(config["langfuse_name"], label=config["label"])
        return prompt
    except Exception:
        # Fallback to local file if Langfuse unavailable
        template = load_prompt_file(version, prompt_type)
        return type("LocalPrompt", (), {
            "compile": lambda self, **kwargs: template.format(**{
                k: v for k, v in kwargs.items()
            }),
            "version": "local",
            "name": config["langfuse_name"]
        })()


def compile_prompt(prompt_type, variables, version=None):
    """Compile prompt with variables. Returns (compiled_text, prompt_object)."""
    prompt   = get_active_prompt(prompt_type, version)
    compiled = prompt.compile(**variables)
    return compiled, prompt


def promote_to_production(version, prompt_type):
    """Promote a staging prompt to production label."""
    registry = load_registry()
    config   = registry[version][prompt_type]

    prompt = langfuse.get_prompt(config["langfuse_name"], label="staging")
    langfuse.create_prompt(
        name=config["langfuse_name"],
        prompt=prompt.prompt,
        labels=["production"],
    )
    print(f"Promoted {version}/{prompt_type} to production")


def list_versions():
    """Show all prompt versions and their deployment status."""
    registry = load_registry()
    active   = registry["active_version"]

    print(f"\nActive version: {active}")
    print(f"{'Version':<8} {'Type':<12} {'Label':<12} {'Deployed':<12} {'Langfuse Name'}")
    print("-" * 70)

    for version, prompts in registry.items():
        if version in ["active_version"]:
            continue
        for ptype, config in prompts.items():
            marker  = " ← active" if version == active else ""
            print(f"{version:<8} {ptype:<12} {config['label']:<12} "
                  f"{str(config['deployed']):<12} {config['langfuse_name']}{marker}")


if __name__ == "__main__":
    print("=== Prompt Version Manager ===\n")
    list_versions()

    print("\nRegistering v2 prompts to Langfuse (staging)...")
    register_version("v2")

    print("\nFinal state:")
    list_versions()

    # Test compilation
    print("\nTest v1 retention prompt:")
    compiled, _ = compile_prompt("retention", {
        "contract": "month-to-month",
        "charges": "$85",
        "tenure": "3 months",
        "probability": "68%"
    }, version="v1")
    print(f"  {compiled}")

    print("\nTest v2 retention prompt:")
    compiled, _ = compile_prompt("retention", {
        "contract": "month-to-month",
        "charges": "$85",
        "tenure": "3 months",
        "probability": "68%"
    }, version="v2")
    print(f"  {compiled}")
