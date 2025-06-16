# GridFM model evaluation


This library is brought to you by the GridFM team to generate power flow data to train machine learning and foundation models.

---

### Workflow

<p align="center">
  <img src="figs/pipeline.png" alt=""/>
  <br/>
</p>


---



### Comparison with other PF datasets/ libraries

| Feature                                                    | GraphNeuralSolver | OPFData | OPFLearn | PowerFlowNet | TypedGNN | PF△ | **GridFM** |
| ---------------------------------------------------------- | ----------------- | ------- | -------- | ------------ | -------- | --- | ---------- |
| Generator Profile                                          | ✅                 | ❌       | ❌        | ✅            | ✅        | ✅   | ❌          |
| N-1                                                        | ❌                 | ✅       | ✅        | ✅            | ✅        | ✅   | ✅          |
| > 1000 Buses                                               | ❌                 | ✅       | ✅        | ❌            | ❌        | ✅   | ✅          |
| N-k, k > 1                                                 | ❌                 | ❌       | ❌        | ❌            | ❌        | ❌   | ✅          |
| Load Scenarios from Real World Data                        | ❌                 | ❌       | ❌        | ❌            | ❌        | ❌   | ✅          |
| Multi-processing and scalable to very large (1M+) datasets | ❌                 | ❌       | ❌        | ❌            | ❌        | ❌   | ✅          |



