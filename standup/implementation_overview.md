# Contingency Analysis Workflow (refer to handwritten version)

## Overview Table

| **Function**          | **Data Generation**                                                                                                                                                                                          | **Model (finetuned on PF)**                                                                                                          | **Post-processing**                                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**           | Generate data with random **n-1** and **n-2** contingencies 🅐                                                                                                                                               | Solve **power flow**                                                                                                                 | - Compute **branch current**  <br> - Check for **branch current violations** (overloading) <br> - Check for **bus voltage violations**                           |
| **Inputs**            | - Aggregated load profiles  <br> - Config file  <br> - Case file                                                                                                                                             | - \( V \) at PV nodes <br> - \( P, Q \) at PQ nodes <br> - \( V, \theta \) at slack <br> - **Scenario index**                        | - **Branch parameters** 🟢<sub>1</sub>  <br> - **Bus parameters** 🟢<sub>2</sub> <br> - **PF solutions** 🟢<sub>3</sub> <br> - **Scenario index** 🟢<sub>4</sub> |
| **Intermediate Data** | - For each scenario: <br> &nbsp;&nbsp;&nbsp;&nbsp;- Node features \( P, Q, \theta \) <br> &nbsp;&nbsp;&nbsp;&nbsp;- Adjacency list <br> &nbsp;&nbsp;&nbsp;&nbsp;- Index of **lines/transformers dropped** 🅑 |                                                                                                                                      |                                                                                                                                                                  |
| **Outputs**           | - **Branch parameters** (admittance and capacity) 🟢<sub>1</sub><br> - **Bus parameters** (voltage bounds and base voltage) 🅒 🟢<sub>2</sub>                                                                | - **PF solutions** 🟢<sub>3</sub> <br> - **Scenario index** 🟢<sub>4</sub> (needed to map scenarios to table of dropped branches) 🅑 | - **Bus voltage violations** <br> - **Branch current violations**                                                                                                |

---

## Remarks

### 🅐 Why We Don’t Generate All Contingencies

We don’t generate all **n-1** or **n-2** contingencies, since that would be too large. For example, the IEEE 300-bus system has around 400 branches. That would require:

400 contingencies × 10,000 load scenarios = 4 × 10⁶ scenarios

Instead, for each load scenario, we generate **20 contingency scenarios**.  
For each contingency scenario, we randomly select one (probability = 0.5) or two branches and drop them.

---

### 🅑 Scenario Indexing and Dropped Branch Tracking

We store the **indices of the branches** that we drop in each scenario.  
This allows us to know which branches we **don’t need to compute the current for** during post-processing.

We also store the **scenario index** with each scenario, so we can map it to the correct indices of dropped branches.

---

### 🅒 Why Additional Parameters Are Needed

We need additional **branch** and **bus parameters** to:

- Compute the branch currents  
- Check for violations

This is why we had to include these in the data generation pipeline.

---
