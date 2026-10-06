# Supporting experiments

These experiments support the project narrative but are not used to redefine the primary baseline after seeing real-event results.

| Experiment | Outcome | Role |
|---|---|---|
| 250k synthetic scaling | Mean synthetic E improved to 0.248501, but real transfer did not consistently improve; CPU training was about 498 min | Scaling ablation |
| Moderate PIML (`lambda=0.01, 0.1`) | ~1% synthetic improvement; each improves 5/6 real event/plane fits | Main physics comparison |
| Strong PIML (`lambda=1`) | Synthetic error worsened and real behavior became strongly event dependent | Negative/regularization-strength result |
| Event-local training | Mixed: helped some planes and hurt others | Exploratory only |
| Multiscale / scale augmentation | Failed or mixed | Retired direction |
| Joint nuisance/ramp prediction | Worse overall | Negative result |
| Robust detrending | Large gains for some events, losses for others; can absorb true deformation | Domain-shift diagnostic only |

The 50k XY5/no-depth model remains the frozen pure-ML baseline.
