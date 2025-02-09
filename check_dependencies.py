"""
Check if all required dependencies are installed.
"""
import sys
import importlib

def check_import(module_name, import_name=None):
    """Check if a module can be imported."""
    try:
        if import_name is None:
            import_name = module_name
        importlib.import_module(import_name)
        print(f"✅ {module_name} is installed")
        return True
    except ImportError as e:
        print(f"❌ {module_name} is NOT installed: {str(e)}")
        return False

# List of (package_name, import_name) tuples
required_packages = [
    ('wikipedia', 'wikipedia'),
    ('numpy', 'numpy'),
    ('networkx', 'networkx'),
    ('sentence-transformers', 'sentence_transformers'),
    ('tqdm', 'tqdm'),
    ('faiss-cpu', 'faiss'),  # faiss-cpu or faiss-gpu imports as 'faiss'
    ('redis', 'redis'),
    ('msgpack', 'msgpack'),
    ('xxhash', 'xxhash'),
    ('scikit-learn', 'sklearn')
]

print(f"Python version: {sys.version}\n")
print("Checking dependencies...")

missing = []
for package_name, import_name in required_packages:
    if not check_import(package_name, import_name):
        missing.append(package_name)

if missing:
    print("\n❌ Missing packages. Install them with:")
    print("pip install " + " ".join(missing))
else:
    print("\n✅ All dependencies are installed!")

# Print package versions for installed packages
print("\nInstalled package versions:")
for package_name, import_name in required_packages:
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"{package_name}: {version}")
    except ImportError:
        print(f"{package_name}: not installed")
