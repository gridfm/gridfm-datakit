# gridfm-datakit


This library is brought to you by the GridFM team to generate power flow data to train machine learning and foundation models.

---

### Workflow

<p align="center">
  <img src="figs/pipeline.png" alt=""/>
  <br/>
</p>


---



### Comparison with other PF datasets/ libraries

| Feature                                                    | GraphNeuralSolver [\[1\]](https://doi.org/10.1016/j.epsr.2020.106547)| OPFData [\[2\]](https://arxiv.org/abs/2406.07234) | OPFLearn [\[3\]](https://arxiv.org/abs/2111.01228)| PowerFlowNet [\[4\]](https://arxiv.org/abs/2311.03415) | TypedGNN [\[5\]](https://doi.org/10.1016/j.engappai.2022.105567)| PF△ [\[6\]](https://www.climatechange.ai/papers/iclr2025/67)| **gridfm-datakit** [\[7\]](https://www.cell.com/joule/fulltext/S2542-4351(24)00470-7) |
| ---------------------------------------------------------- | ----------------- | ------- | -------- | ------------ | -------- | --- | ---------- |
| Generator Profile                                          | ✅                | ❌      | ❌       | ✅           | ✅       | ✅  | ❌         |
| N-1                                                        | ❌                | ✅      | ✅       | ✅           | ✅       | ✅  | ✅         |
| > 1000 Buses                                               | ❌                | ✅      | ✅       | ❌           | ❌       | ✅  | ✅         |
| N-k, k > 1                                                 | ❌                | ❌      | ❌       | ❌           | ❌       | ❌  | ✅         |
| Load Scenarios from Real World Data                        | ❌                | ❌      | ❌       | ❌           | ❌       | ❌  | ✅         |
| Multi-processing and scalable to very large (1M+) datasets | ❌                | ❌      | ❌       | ❌           | ❌       | ❌  | ✅         |