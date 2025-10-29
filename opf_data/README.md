
This script converts OPF-data JSON files into four CSV outputs structured like those of datakit (branch_data.parquet, bus_data.parquet, gen_data.parquet, and y_bus_data.parquet).
It processes input files in parallel and in fixed-size chunks, appending sorted results incrementally to the final output datasets.

### Example usage

```bash
python batch_convert.py path_to_opf_data_dir/ out_dir/ "case57_ieee" --n-1 True
```

### Required Arguments

- `data_dir`: Directory containing example*.json files (searched recursively)
- `out_dir`: Directory for aggregated CSV outputs
- `network`: Network name (without pglib prefix)
- `--n-1`: Boolean flag indicating whether using N-1 contingency analysis (True) or full topology (False)

### Optional Arguments

- `--chunk-size`: Files per processing chunk (default: 20000)
- `--atol`: Absolute tolerance for Ybus equality (default: 1e-5)
- `--rtol`: Relative tolerance for Ybus equality (default: 1e-5)

## 1. Fields Not Used

### **Global / Metadata**

* `metadata.objective` – not written to outputs (used only for a cost check).
* `grid.context` – only the first scalar `baseMVA` is used; all other entries ignored.

### **Bus Data**

* No unused numerical fields beyond the inactive bus type (type 4), which is rejected.

### **Generator Data**

* `pg` (initial active power)
* `qg` (initial reactive power)
* `vg` (initial voltage magnitude)
* `mbase` (not output; only compared to `baseMVA`)

### **Load Links**

* `edges.load_link.senders` – used only for length checks, not output or computation.

### **Generator Links**

* `edges.generator_link.senders` – used only for length checks.

### **Shunt Links**

* `edges.shunt_link.senders` – used only for length checks.

### **AC Lines**

* `rate_b`, `rate_c` – short-term and emergency ratings ignored.

### **Transformers**

* `rate_b`, `rate_c` – unused.
* `shift` – asserted to be zero; not used.
* `rate_a` is used, but `tap`, `angmin`, and `angmax` only contribute to admittance and limits, not to flows beyond that.

---

## 2. Assumptions and Assertions

### **Physical / Network Structure**

* `br_x != 0.0` for all branches and transformers (no zero reactance).
* `tap_mag > 0.0` for all transformers (no zero tap).
* `shift == 0.0` and `b_fr == b_to == 0.0` for transformers.
* `b_fr == b_to` for AC lines (symmetric shunt susceptance).
* All index references (`senders`, `receivers`) are within the bus count bounds.
* Bus type must be one of {1, 2, 3}; type 4 (“inactive”) triggers an error.
* Generator `mbase` must equal global `baseMVA`.
* The computed generator quadratic cost function
  [
  \sum_k (c0 + c1·p_g + c2·p_g^2)
  ]
  must match `metadata.objective` within tolerance (1e-6).
* The computed admittance matrix (Y) must match the baseline within
  absolute tolerance 1e-9 and relative tolerance 1e-9.

---

## 3. Constant Values in Output Data

### **`branch_data.parquet`**

| Column                | Constant Value | Context                                 |
| --------------------- | -------------- | --------------------------------------- |
| `tap` (AC lines only) | 1.0            | AC lines have no tap ratio.             |
| `br_status`           | 1.0            | All branches are considered in service. |

### **`gen_data.parquet`**

| Column        | Constant Value | Context                        |
| ------------- | -------------- | ------------------------------ |
| `et`          | `"gen"`        | Static label identifying type. |
| `is_gen`      | 1              | All rows are generators.       |
| `is_sgen`     | 0              | No static generators.          |
| `is_ext_grid` | 0              | No external grid sources.      |
| `in_service`  | 1              | All generators active.         |
