"""
Test data generation consistency.

This test generates data using a test config and compares it against
reference data to ensure column structure and content remain consistent.
"""

import pytest
import pandas as pd
from pathlib import Path
import numpy as np
import tempfile
import yaml
from gridfm_datakit.generate import generate_power_flow_data_distributed
from gridfm_datakit.utils.param_handler import NestedNamespace


class TestDataGenerationConsistency:
    """Test that generated data maintains consistency with reference data."""

    @pytest.fixture
    def test_config(self):
        """Load the test configuration."""
        config_path = Path(__file__).parent / "config" / "consistency_test.yaml"
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return NestedNamespace(**config_dict)

    @pytest.fixture
    def reference_data_dir(self):
        """Get path to reference data directory."""
        # Using new_data as reference (old generation)
        ref_dir = Path(__file__).parent / "reference_data"
        assert ref_dir.exists(), f"Reference data directory not found at {ref_dir}"
        return ref_dir

    @pytest.fixture(scope="session")
    def generated_data_dir(self):
        """Generate data in a temporary directory (once per session)."""
        config_path = Path(__file__).parent / "config" / "consistency_test.yaml"
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        test_config = NestedNamespace(**config_dict)

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_config.settings.data_dir = tmp_dir
            test_config.settings.overwrite = True

            print(f"\n{'=' * 80}")
            print(f"Generating test data in: {tmp_dir}")
            print(f"{'=' * 80}\n")

            # Generate data
            generate_power_flow_data_distributed(test_config)

            yield Path(tmp_dir)

    @staticmethod
    def get_parquet_files(directory):
        """Get all parquet files in directory."""
        parquet_files = {}
        for file in directory.glob("**/*.parquet"):
            parquet_files[file.name] = file
        return parquet_files

    @staticmethod
    def compare_column_content(df1, df2, col):
        """Compare content of a specific column between two dataframes."""
        try:
            # Check if lengths match
            if len(df1) != len(df2):
                return {
                    "status": "LENGTH_MISMATCH",
                    "len_dir1": len(df1),
                    "len_dir2": len(df2),
                }

            # Compare types
            if df1[col].dtype != df2[col].dtype:
                return {
                    "status": "TYPE_MISMATCH",
                    "type_dir1": str(df1[col].dtype),
                    "type_dir2": str(df2[col].dtype),
                }

            # Check if values are equal
            are_equal = np.allclose(df1[col], df2[col])

            if are_equal:
                return {"status": "IDENTICAL"}
            else:
                differences = (df1[col] != df2[col]).sum()
                return {
                    "status": "VALUES_DIFFER",
                    "num_differences": differences,
                    "percent_different": (differences / len(df1)) * 100,
                }
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
            }

    def test_column_content_consistency(self, reference_data_dir, generated_data_dir):
        """Test that common columns have identical content."""
        print(f"\n{'=' * 80}")
        print("TEST: Column Content Consistency")
        print(f"{'=' * 80}\n")

        ref_files = self.get_parquet_files(reference_data_dir)
        gen_files = self.get_parquet_files(generated_data_dir)

        # Find matching files
        common_files = set(ref_files.keys()) & set(gen_files.keys())

        for filename in sorted(common_files):
            print(f"  Checking {filename}...")
            df_ref = pd.read_parquet(ref_files[filename])
            df_gen = pd.read_parquet(gen_files[filename])

            ref_cols = set(df_ref.columns)
            gen_cols = set(df_gen.columns)

            # Get common columns (excluding the new load_scenario_idx)
            common_cols = sorted((ref_cols & gen_cols) - {"load_scenario_idx"})

            print(f"    Comparing {len(common_cols)} columns...")

            # Check each common column
            for col in common_cols:
                comp_result = self.compare_column_content(df_ref, df_gen, col)

                if comp_result["status"] != "IDENTICAL":
                    print(f"      ✗ {col}: {comp_result}")

                assert comp_result["status"] == "IDENTICAL", (
                    f"Column '{col}' in {filename} differs: {comp_result}"
                )

            print(f"    ✓ All {len(common_cols)} columns identical\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
