import yaml

def load_yaml(file_path):
    """Load a YAML file and return its contents."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def compare_dicts(d1, d2, path=""):
    """Recursively compare two dictionaries."""
    differences = []
    
    # Check all keys in d1
    for key in d1:
        current_path = f"{path}.{key}" if path else key
        
        if key not in d2:
            differences.append(f"KEY MISSING in File 2: {current_path}")
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            # Recursively compare nested dictionaries
            differences.extend(compare_dicts(d1[key], d2[key], current_path))
        elif d1[key] != d2[key]:
            differences.append(
                f"VALUE CHANGED at {current_path}:\n"
                f"  File 1: {d1[key]}\n"
                f"  File 2: {d2[key]}"
            )
    
    # Check for keys in d2 that aren't in d1
    for key in d2:
        current_path = f"{path}.{key}" if path else key
        if key not in d1:
            differences.append(f"KEY ADDED in File 2: {current_path}")
    
    return differences

def compare_yaml_files(file1, file2):
    """Compare two YAML files and print differences."""
    yaml1 = load_yaml(file1)
    yaml2 = load_yaml(file2)
    
    print("=" * 60)
    print(f"Comparing:\n  File 1: {file1}\n  File 2: {file2}")
    print("=" * 60)
    
    differences = compare_dicts(yaml1, yaml2)
    
    if not differences:
        print("\n✓ Files are identical!")
    else:
        print(f"\n✗ Found {len(differences)} difference(s):\n")
        for diff in differences:
            print(diff)
            print()
    
    return yaml1, yaml2, differences

# Example usage
if __name__ == "__main__":
    file1 = "/u/apu/gridfm-datakit/scripts/config/default_pf.yaml"
    file2 = "/u/apu/gridfm-datakit/scripts/config/matteo_config.yaml"
    compare_yaml_files(file1, file2)
    