# Visual response to Jorge's clarification

Jorge clarified that his request concerned heterogeneous input/output images with accompanying
interpretation. It did not request another numerical or ordinal rating instrument.

## Figure contract

Delivery surface: the compiled thesis PDF, in the Results chapter. Renderer: Matplotlib PDF with
native RGB image arrays, lossless compression and no pixel resampling. Each detail case occupies
one PDF page with two directly labelled columns, a white background, dark text and quiet grey
borders. Colour does not encode a verdict. Each row uses identical pixel coordinates and display
size for all methods; the retained LR nearest-neighbour image exposes the input without an added
enhancement. The gallery compares regions, not rescaled metrics or different experiments.

| Figure | Question and evidence | Intended interpretation |
| --- | --- | --- |
| `pretrained-detail-gallery.png` | Three previously fixed SMB cases: x2 clean, x2 strong, x4 strong; HR, LR, EDSR, SwinIR | Contrast a reconstruction without a clear reviewed issue with altered texture/contours and losses in fine notation/text. |
| `adapted-detail-gallery.png` | Two fixed adaptation cases, x4 moderate and strong; HR, LR, official and adapted EDSR | Separate clearer geometry and reduced halos from information that the adapted model still cannot recover. |
| `external-detail-gallery.png` | Three fixed external cases covering accepted, accepted with reservations and rejected decisions | Make professional limitations inspectable alongside the twelve complete-page comparisons. The decision is for the full reviewed page, not for this crop alone. |

The qualitative section on pretrained methods now precedes the adaptation section, so the reader
sees baseline limitations before the improvement from adaptation. Existing complete pages and
larger contextual crops remain. Each new gallery has a caption, source attribution and a nearby
paragraph identifying what to inspect.

The cases were fixed for the original reviews. Detail regions were chosen afterwards for this
editorial explanation and their coordinates are recorded in `detail_regions` in the figure
manifest. PNG contact sheets are retained only as previews; the thesis includes the native-image
PDFs. They are not claimed as a newly preregistered sample, a new human review, or a frequency
estimate. No model inference, degradation calibration, weights, metrics or reviewer decisions
were changed.

## Reproduction and final-context checks

Run `uv run python scripts/generate_thesis_qualitative_figures.py` from the project root. The
manifest records 21 composites: eight analytical-crop figures (including these three galleries)
and thirteen full-page comparisons (one SMB plus all twelve external cases).

Verification includes input/output hashes, exact source-to-thesis file copies, aligned crop
bounds, retained review identities, Ruff, relevant evidence tests, LaTeX compilation, inspection
of the galleries in the rendered PDF, and PDF/A validation. Build outputs remain local; Overleaf
is the final compiler.

Initial verification on 7 September 2026, before the subsequent native-resolution correction:
all 21 input/output and thesis-copy hashes reconciled;
eight detail regions have valid bounds and the SMB cases resolve to the retained reviews.
Ruff passes and all 27 relevant evidence/governance tests pass. The coverless PDF compiles to
92 pages with no unresolved references/citations or overfull boxes, and veraPDF reports
`PASS ... 2b`. Figures 6.4, 6.10 and 6.14 were inspected in the final PDF (printed pages 34,
39 and 45; physical PDF pages 46, 51 and 57). Section boundaries flush pending figures so they
remain with the corresponding experimental stage. The twelve complete external pages remain.

## Native-resolution correction after reader feedback

The former 300 dpi PNG mosaics resampled complete pages before LaTeX embedded them. Their small
printed size also obscured detail. All 21 composites now have a publication PDF that embeds the
original RGB arrays losslessly (`interpolation="none"`); every thesis reference uses that PDF.
Detail galleries have one case per page in a two-column layout, doubling panel width relative to
the old four-column contact sheets. Full-page views retain their original pixel dimensions for
digital zoom. PNG outputs are previews only. Each publication PDF has a separate hash in the
manifest; three files are multipage galleries containing eight detail cases in total.

Validation: eight embedded adaptation crops extracted from the publication PDF equal the original
RGB crops exactly after accounting for the PDF storage Y-axis orientation. The four crops
extracted from physical page 54 of the final thesis also match exactly. All 21 publication copies
and source hashes reconcile; Ruff and 18 relevant tests pass. The final 100-page PDF has no
unresolved references or overfull boxes and passes veraPDF PDF/A-2b. Printed pages 42, 43 and 50
(physical pages 54, 55 and 62) were visually inspected for orientation, labels and readable detail.
The larger file (approximately 75 MB) reflects preservation of the full source image resolution.
