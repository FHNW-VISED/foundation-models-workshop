# Glossary: foundation-models-workshop

Terms for running the AMLD 2027 workshop when the GPU platform is uncertain (#37, #39).

**Fallback ladder**: the steps a participant follows on the day when the platform fails, in order: Colab with a T4 → Colab CPU runtime → pair with a neighbour → HTML fallback. Lives in FACILITATION.md and on the first slide.

**CPU check**: one timed, full (non-smoke) run of the notebook on a Colab CPU runtime. Passes if every section takes at most twice its minute budget and a SAM click in the playground takes at most 3 s. Decides whether the ladder's CPU step exists.

**Decision gate**: a dated go/no-go point for the platform: 2027-01-15 (answer and platform decision), 2027-02-10 (an external test login runs the whole notebook), 2027-02-24 (dry run on the chosen platform, then frozen).

**HTML fallback**: the offline export of the executed notebook, one file per section plus an index, shipped as a zip with the setup material. The last step of the ladder, for reading along only.

**Embedded control**: a setting whose every position is precomputed and can be switched in the HTML fallback.

**Frozen control**: a setting the HTML fallback shows only at its executed value, marked "Frozen in this copy: open the notebook in Colab to try it."
