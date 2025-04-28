
# Progress

|                             | Data Gen                                               | PF post-processing                                                                                     | Model                                                                                                                 | Post processing on model outputs                                                       | Violation detection                                                                                |
| --------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| To Implement TODAY<br>04/28 | [ ] Change current code to run PF after removing lines | [ ] Add line capacity (current or apparent power)<br><br>[ ] Add information on which line was dropped |                                                                                                                       | [ ] Compute power flows from bus-level results (need to implement a function for that) |                                                                                                    |
| For later                   |                                                        |                                                                                                        | [ ]  Test model finetuned on PF on case 300<br>[ ] Re-training on new data (since new data shows larger line loading) | [ ] Compare the results of the function against pandapower                             | Check for overloading (current/ apparent power) and voltage violation and come up with a criterion |
|                             |                                                        |                                                                                                        |                                                                                                                       |                                                                                        |                                                                                                    |
# 04/28

## Meeting Agenda
- show contingency analysis features from pandapower -> show notebook
	- https://pandapower.readthedocs.io/en/latest/contingency.html
	- Checks voltage magnitude and line overload
- Quick presentation of the code to change 
- Quick presentation of the pipeline 
## "Theory"

- [ ] Figure out what to do when AC-PF doesn't converge (we can't label the data in that case)
	- [ ] Solution for now: discard them
> 	"in _steady-state power flow_, "no convergence" usually means that the solver can't find a physically realistic equilibrium given the conditions."
    
> 	"it doesn't necessarily mean that the system immediately "blows up" in real life — but it very likely indicates that the post-contingency steady-state is **unreachable** without corrective actions (e.g., load shedding, generator rescheduling, remedial actions)."
	    
> 	"In **real-world dynamics**, if no action is taken fast enough, the system could **become unstable** (voltage collapse, angle instability, cascading failures)"

- [ ] Figure out whether to use apparent power or current to check for overloading

- [ ] Compare to DC
	- How to define line overloading with DC since no reactive power?

- [ ] Compare to AC with less iterations
	- Will that really speed up things?


 

