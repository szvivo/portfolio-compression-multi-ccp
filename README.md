# Portfolio compression in financial networks: a multi-CCP model

Simulation code for the BA thesis *Portfolio compression in financial networks: a multi-CCP model*, Corvinus University of Budapest, 2026.

The thesis asks what happens to portfolio compression when clearing is divided among several central counterparties that cannot see each other's obligations. This repository contains the complete simulation behind Chapter 5: it generates synthetic obligation networks, clears them under seven different arrangements, settles them with the Eisenberg–Noe clearing vector, and reports gross notional reduction, unpaid claims and defaults.

## Contents

| File | Description |
|---|---|
| `thesis_simulation.ipynb` | The complete simulation. Running it end to end reproduces every table and figure in Chapter 5. |
| `thesis_model.py` | The model itself: network generation, compression, clearing and the shock experiment. Written out by the notebook; included here so it can be read without running anything. |
| `requirements.txt` | Package versions. |

## Running it

```bash
pip install -r requirements.txt
jupyter notebook thesis_simulation.ipynb
```

Then *Kernel → Restart & Run All*. The full run takes about a minute and writes `OUTPUT.xlsx`, a workbook with one sheet per result table.

## What the model does

Obligation networks are generated in two topologies: a random network in which every possible obligation exists with fixed probability, and a core-periphery network in which a small set of dealers trades heavily with each other and with everyone else. Obligations are carried in four asset classes, drawn independently, so the same pair of institutions may be connected in several classes at once.

Compression is solved exactly, as a linear programme that minimises gross notional subject to three constraints: every net position is unchanged, no obligation is increased, and no new counterparty relationship is created. This is the conservative case of D'Errico and Roukny (2021). Settlement uses the Eisenberg and Noe (2001) clearing vector, computed once on the post-netting network.

The experiment is organised into four blocks:

- **A** — gross notional reduction under each clearing arrangement, across market sizes and both topologies
- **B** — the effect of moving asset classes to dedicated counterparties
- **C** — the effect of liquidity stress, sweeping outside assets from zero upwards
- **D** — damage caused by a shock, varying which institutions are hit

Two properties are verified after every compression: net positions are unchanged, and no counterparty relationship exists that did not exist before. Three validation tests are run before any results are generated, each against a result that exists independently of the code.

## Parameters

Networks are synthetic. The parameters were chosen to produce workable connectivity rather than to match any particular market, so the magnitudes reported should be read as indicative and the direction of each comparison as the result. All parameters are listed in the notebook and in the appendix of the thesis.

## References

Craig, B. R., & von Peter, G. (2010). *Interbank tiering and money center banks* (Discussion Paper Series 2: Banking and Financial Studies No. 12/2010). Deutsche Bundesbank.

D'Errico, M., & Roukny, T. (2021). Compressing over-the-counter markets. *Operations Research, 69*(6), 1660–1679.

Duffie, D., & Zhu, H. (2011). Does a central clearing counterparty reduce counterparty risk? *The Review of Asset Pricing Studies, 1*(1), 74–95.

Eisenberg, L., & Noe, T. H. (2001). Systemic risk in financial systems. *Management Science, 47*(2), 236–249.

## Licence

MIT. See `LICENSE`.
