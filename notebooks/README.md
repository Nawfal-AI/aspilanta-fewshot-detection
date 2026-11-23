# Notebooks

The original research notebooks, kept as the record of how the work was actually
done. The package in `src/aspilanta/` is the maintained version — see
[`docs/code_map.md`](../docs/code_map.md) for which cell became which module.

## Files

| File | Cells | Outputs | What it is |
|---|---|---|---|
| `00-research-notebook-final.ipynb` | 164 | **kept** | The canonical notebook. All 11 sections. |
| `archive/01-full-runall.ipynb` | 125 | stripped | Earlier full run |
| `archive/02-main-solution.ipynb` | 120 | stripped | Earlier draft with the main solution |
| `archive/03-new-notebook.ipynb` | 125 | stripped | Earlier draft |

**Only the canonical notebook has the complete work.** The three archived
versions predate the ablation studies, the cross-family baselines, and the
Section 11 corrections. They are here for provenance, not for use.

Outputs are kept in the canonical notebook because they *are* the project's
evidence — the training curves, confusion matrices and result tables were read
off those cells. Outputs are stripped from the archives, which took them from
roughly 11 MB each to under 450 KB.

## A note on the source files

The upload contained five notebooks, but two of them
(`full-runall__3_.ipynb` and `full-runall__4_.ipynb`) were **byte-identical in
code** — they differed only in stored outputs. Only one is kept here, as
`archive/01-full-runall.ipynb`.

## Structure of the canonical notebook

| Section | Contents |
|---|---|
| 0 | Environment setup and weight restore |
| 1 | Base dataset preparation, auxiliary YOLOv8n baseline |
| 2 | Synthetic data generation |
| **3** | **The proposed solution** |
| 4 | Main experiments 1, 2, 3 |
| 5 | Deployment evaluation |
| 6 | Ablation studies (nine runs) |
| 7 | Foundation-model baseline: OWL-ViT |
| 8 | Model-oriented baseline: TFA |
| 9 | Algorithm-oriented baseline: prototypical network |
| 10 | Final archive |
| 11 | Corrections, held-out test split, end-to-end validation |

Sections 3 onward are the contribution; 0 to 2 prepare its inputs.

## Running them

The notebooks were written for Kaggle and contain `/kaggle/working` and
`/kaggle/input` paths throughout, plus cells that restore weights from attached
Kaggle datasets. They will not run unmodified elsewhere.

Use the package instead:

```bash
python scripts/train.py --stage all
```

Or, from a notebook of your own:

```python
import sys; sys.path.insert(0, "src")
from aspilanta import AspilantaDetector, load_config

config = load_config("configs/kaggle.yaml")
detector = AspilantaDetector(config=config.deployment)
```

## If you edit these

Strip outputs from the archives before committing:

```bash
make strip-notebooks
```

Or install the git filter once, so it happens automatically:

```bash
pip install nbstripout && nbstripout --install
```

Note that this would also strip the canonical notebook, which you probably do
not want. `.gitattributes` marks notebooks as documentation so they do not
skew GitHub's language statistics.
