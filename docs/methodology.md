# Methodology

How the pipeline works, and why each piece is there.

---

## The problem

A detector for a new pest species normally needs thousands of labelled
examples. For *Aspilanta* there were 55 real photographs in total, 44 of them
usable as cutouts, and most were close-up specimen shots rather than the
cluttered trap-card images the detector must work on at deployment. So there
are two gaps, not one:

1. **Quantity.** 44 examples against the thousands a detector expects.
2. **Domain.** The examples that exist look nothing like the deployment input:
   a 5 mm moth among dozens of other insects on a glossy adhesive card.

The second gap is the harder one. Collecting a few hundred more specimen
photographs would not close it.

---

## Three families, three techniques

The few-shot object detection literature splits into three families, and none
of them solves this problem alone:

- **Data-oriented** methods manufacture labelled examples. They produce images,
  not a detector.
- **Model-oriented** methods use scarce labels more efficiently through
  transfer learning. Efficiency is not creation: with 44 examples there is
  not enough to be efficient with.
- **Algorithm-oriented** methods classify from a handful of references and do
  it well. But they classify *regions*; they cannot find them in a full image.

The method takes one technique from each, arranged so that each covers the
others' gap.

---

## 1. Data-oriented: copy-paste compositing

The generator turns 44 cutouts into 2,000 trap images containing 9,211
labelled instances. Naive alpha compositing would not work: a detector
trained on obvious paste seams learns the seams. Six things make the output
usable.

**Physical scale calibration.** Insect length is drawn from a Gaussian
truncated to 2.5–8 mm and converted to pixels through the detected pad width
(a standard trap card is 140 mm across). A 5 mm moth is therefore 5 mm whether
the background photograph was shot at 1000 or 4000 px wide. Without this, apparent
size would encode nothing but source resolution.

**Pad-aware placement.** Insects are composited only onto the detected
adhesive pad, not anywhere on the photograph. This turned out to be the
single largest compositing effect in the ablations (−0.269 AP@50 when removed).

**Pad cleaning.** Background photographs are real trap cards, so most already
carry insects. Left in, those become unlabelled positives: objects with no
ground-truth box, teaching the detector to suppress exactly what it should
find. They are detected as small dark blobs against the smooth pad and
inpainted away.

**Poisson blending.** `cv2.seamlessClone` solves for pixel values whose
gradients match the source inside the mask while matching the destination at
the boundary, so the seam disappears even when the two images were lit
differently.

**Per-instance appearance variation.** Colour jitter, local lighting match,
contact shadow, occasional motion blur. Without these, every instance descends
from one of 44 cutouts and shares its colour statistics, a shortcut cue.

**Distractors.** European moth species composited alongside the target, so the
detector must discriminate rather than just detect "an insect". (The ablation
did not support this one; see `results.md`.)

Labels are validated before writing: a box that has drifted off-frame or
collapsed to a sliver is rejected and counted, rather than becoming quiet
training noise.

## 2. Model-oriented: staged transfer learning

YOLOv8s, in two stages.

**Stage 1** trains on the three data-rich classes of the Yellow Sticky Traps
dataset (MP, NT, WF) from ImageNet initialisation. *Aspilanta* is absent
entirely. This teaches the network the visual domain: small insects on
adhesive cards, using real photographs and real labels.

**Stage 2** fine-tunes to four classes on the original data plus real and
synthetic *Aspilanta*, **with the backbone trainable**.

That last choice contradicts standard practice and contradicted this project's
own original design. Freezing the backbone is the usual defence against
few-shot overfitting, but by the time stage 2 begins the generator has already
supplied 2,000 images and 9,211 instances; the scarcity the freeze protects
against no longer exists. The ablation confirmed it on both validation sets and
the method changed accordingly.

YOLOv8s rather than YOLOv8n: the size ablation measures the difference
directly, and 's' wins on small insects at essentially the same inference cost.

## 3. Algorithm-oriented: frozen CLIP prototype filter

Embed a few real *Aspilanta* crops with a frozen CLIP encoder, average them
into a prototype, and re-score candidate detections by cosine similarity.
Nothing is trained; the few-shot signal enters entirely through the support
set.

**This is off by default**, on measured evidence: applied to every detection it
costs more recall than it buys precision. Section 11.3 recovers part of it by
consulting CLIP only for detections the detector itself is uncertain about.

The similarity threshold is calibrated from the support set as
`mean − std` of support-to-prototype similarity, rather than hand-picked -
deliberately conservative and reproducible.

---

## Deployment

Single-pass inference on the whole image, the target class reported alone.
MP/NT/WF are still detected internally and appear in the returned DataFrame as
context, but they are not drawn or counted: the deployment task is an
*Aspilanta* count.

Tiled inference is implemented and off. On this data it scores F1 0.023 against
0.861 for single-pass: the detector was trained on whole images resized to
640 px, so a native-resolution 640 px crop presents insects at a scale it has
never seen. The code is kept because the failure is a clean illustration of
train/inference domain mismatch, and because the motivation is sound for
genuinely enormous images where a 5 mm insect would survive a resize as two
pixels. Re-measure before trusting it on your own data.

---

## Evaluation design

Three experiments on one common validation set, so the numbers are comparable:

1. **Base detector** on the three-class split, establishing the domain is
   learnable at all.
2. **Zero-shot diagnostic**: the three-class model run over trap images that
   contain *Aspilanta*. It has no output for the class, so every detection is
   wrong by construction. This is the control that makes the domain gap a
   measurement rather than an assumption.
3. **Proposed detector** on the four-class set.

Nine ablations then remove one component at a time. Two families, scored
differently: ablations that change stage 2 are evaluated on the four-class
target-domain set; ablations that change stage 1 are evaluated on the
three-class base set, because a three-class model has no target-class output to
score.

Three cross-family baselines (OWL-ViT, TFA, prototypical network) check that
the pipeline beats the obvious alternatives, not just nothing.

Finally, a decision-held-out test split. The proposed configuration was chosen
partly by reading ablation numbers off the validation set, which means that set
can no longer support an unbiased final claim. The split is stratified on
provenance so both halves keep the same real/synthetic mix, with only eleven
real photographs, an unstratified draw could put nearly all of them on one side.
