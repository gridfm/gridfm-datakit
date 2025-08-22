# gridfm-datakit


This library is brought to you by the GridFM team to generate power flow data to train machine learning and foundation models.

---

### Workflow

<p align="center">
  <img src="https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main/docs/figs/pipeline_docs.png" alt=""/>
  <br/>
</p>


---



### Comparison with other PF datasets/ libraries
<div style="display: flex; justify-content: center;">
  <div style="transform: scale(0.75); transform-origin: top center;">
    <table>
      <thead>
        <tr>
          <th>Feature</th>
          <th><a href="https://doi.org/10.1016/j.epsr.2020.106547">Graph Neural Solver [1]</a></th>
          <th><a href="https://arxiv.org/abs/2406.07234">OPFData [2]</a></th>
          <th><a href="https://arxiv.org/abs/2111.01228">OPFLearn [3]</a></th>
          <th><a href="https://arxiv.org/abs/2311.03415">PowerFlowNet [4]</a></th>
          <th><a href="https://doi.org/10.1016/j.engappai.2022.105567">TypedGNN [5]</a></th>
          <th><a href="https://www.climatechange.ai/papers/iclr2025/67">PF△ [6]</a></th>
          <th><a href="https://openreview.net/pdf?id=cecIf0CKnH"><strong>PGLearn [7]</strong></a></th>
          <th><strong><a href="https://www.cell.com/joule/fulltext/S2542-4351(24)00470-7">gridfm-datakit [8]</a></strong></th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Generator Profile</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>✅</td></tr>
        <tr><td>N-1</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td></tr>
        <tr><td>&gt; 1000 Buses</td><td>❌</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td></tr>
        <tr><td>N-k, k &gt; 1</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td></tr>
        <tr><td>Load Scenarios from Real World</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td></tr>
        <tr><td>Net Param Perturbation</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td></tr>
        <tr><td>Scalable to 1M+</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td></tr>
      </tbody>
    </table>

  </div>
</div>
