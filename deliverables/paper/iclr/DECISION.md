# Read this first — the abstract is due today

**ICLR 2027 abstract deadline: Sep 18, 2026, 11:59 PM AOE** (= Sep 19, 11:59 UTC).
**Full paper: Sep 25, 2026, 11:59 PM AOE.** The guidelines say the deadlines are
final with no accommodation for missing either.

## My recommendation: register the abstract, with one condition

Registering is close to free — **but not completely free**, and the reason is the
one thing I got wrong when I first advised you.

### The irreversible part

> "No authors can be added or removed after the abstract submission deadline.
> (Author order can be changed after the abstract submission deadline up to the
> full paper deadline.)" — ICLR 2027 Author Guidelines
>
> "We make no exceptions whatsoever."

**The author set is permanent at tonight's deadline.** Order can still change
until Sep 25; membership cannot, ever. So the one decision you cannot defer is
*who is on this paper*. If there is any chance a collaborator should be on it —
someone who might run raters, verify licensing, or co-write §7 — **add them
tonight**. You can reorder later; you can never add.

### The reversible part

> "Before the paper submission deadline, if an author withdraws the paper it
> will be deleted from the OpenReview hosting site." — same document

Withdrawing before **Sep 25** deletes the submission entirely: never public, no
record, no penalty. The guidelines explicitly bless registering an abstract you
may not use. Withdrawing *after* Sep 25 is a different thing: the submission
stays permanently public **and is de-anonymized immediately**.

So: **the free exit closes Sep 25, not tonight.** Register, then decide on the
25th with a week of data in hand.

### The one thing that could desk-reject you regardless

**Reciprocal reviewing.** Every submission needs at least one author registered
to review ≥3 papers, and eligibility is judged **as of the abstract deadline**.
If no author is registered, it is a desk reject. Qualification requires a prior
accepted paper at ICLR/NeurIPS/ICML/CVPR/ACL-family/TMLR and several others.
**Check this tonight** — it is not something you can fix on the 25th.

Also: all co-authors need OpenReview profiles by the *full paper* deadline, and
a new profile without an institutional email can take **two weeks** to clear
moderation. Another reason to name everyone tonight.

---

## What decides the 25th

Not the writing — that is in reasonable shape. Two things:

1. **Can you get 8+ naive raters through the study?** See `RUNBOOK.md §0` first;
   there is a design problem with the central pair that you should fix or scope
   before collecting.
2. **Can the main text reach 9 pages?** It is currently 13. This is a
   desk-reject condition, not a reviewer complaint: *"Papers with main text
   beyond the page limit will be desk-rejected."* References and appendix are
   unlimited and do not count, so this is a moving problem, not a cutting one.

If either fails, withdraw on the 24th and take it to **NeurIPS Evaluations &
Datasets** (the renamed D&B track; ~May 2027 deadline based on 2026's). That
track lists "Evaluation Methodology and Metrics" as a contribution type, and its
reviewer guidelines ask literally: *"Does it shift conclusions if applied to
existing benchmarks?"* — which is what §6 is.

**CVPR 2027 is not a usable fallback.** Its deadline is Nov 16, 2026, a month
before ICLR decisions (Dec 16), and its dual-submission rule bars anything under
review elsewhere. Using it would mean withdrawing from ICLR around Nov 10 — by
which point withdrawal is permanently public and de-anonymized.

---

## Done tonight, relevant to the submission

- **The AI use statement now exists** (`sec/9_statements.tex`). It is
  **mandatory** at ICLR 2027 and did not exist in the draft. It does not count
  against the page limit. **Read it and correct it if it misstates anything** —
  the guidelines make a misrepresentation produced by an LLM a Code of Ethics
  violation and grounds for desk rejection, so this section has to be right.
- A reproducibility statement was added to the main text (also off-limit; the
  appendix section alone did not satisfy the convention).
- ICLR 2027 has **no paper checklist and no reproducibility checklist** — I
  confirmed this against both the author and reviewer guidelines. NeurIPS does,
  and its Q7 requires error bars; the suite already emits them.

## Section headings and title — checked, no change needed

Declarative section headings are accepted at ICLR (*LLMs Cannot Self-Correct
Reasoning Yet*, ICLR 2024 oral) but are rare at top level; the closest-genre
paper, *The Role of ImageNet Classes in FID* (ICLR 2023), uses sober noun
phrases and questions. The headings are now noun phrases with claims at
subsection level, which matches. First-level headings render in 12pt small caps,
so long ones are punished — all current ones are short.

The title — declarative claim + colon subtitle — is the plurality style (46% of
NeurIPS 2024) and over-represented among award winners. Keep it.
