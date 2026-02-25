# Generative Uncertainty for Molecular Generation — One-Page Summary

## What we did
We took a method from image generation (Jazbec et al., 2025) that estimates how *confident* a diffusion model is about each sample it generates, and applied it to molecular graph generation (GDSS, ZINC250k). The idea: generate many versions of the same molecule by slightly varying the model's parameters. If the results are consistent, the model is confident. If they vary wildly, it's uncertain — and uncertain molecules are probably bad. Remove them, keep the confident ones, get a better set.

## What we found
**It doesn't work.** Filtering molecules by uncertainty never improves quality over random selection. In fact, it makes things slightly worse.

## Why — three intuitive reasons

**1. The safety net catches everything.**
GDSS has a built-in correction step (Langevin corrector + valence correction) that forces every molecule to be chemically valid. It's like spell-check that fixes every typo automatically. If every molecule is already valid, there are no bad ones to filter out. The most "uncertain" molecules are actually the most *unusual* ones — and removing the unusual ones just makes the set more boring, not better.

**2. Molecules are digital, images are analog.**
An image is continuous — you can make a pixel 1% brighter and get a slightly different image. A molecule is discrete — a bond is either single, double, or triple, nothing in between. When we slightly perturb the model's parameters, the continuous bond probabilities change a little, but the final discrete bond type (picked by argmax) almost never flips. It's like adjusting the volume knob by 0.1% — the music is technically different, but you can't hear it. The discretization step destroys the uncertainty signal before we can use it.

**3. When perturbations are big enough to matter, everything breaks.**
We accidentally tested this: a bug made the perturbations ~100x too large, and 99.5% of molecules became garbage. There's no sweet spot — either the perturbation is too small to change the discrete output, or it's large enough to break molecules entirely. With images, there's a smooth gradient from "slightly different" to "very different." With molecules, there's a cliff.

## What does work
When we deliberately make generation harder (fewer diffusion steps, no correction), uncertainty *does* predict which molecules will be valid vs invalid. At 50 steps without correction (53% validity), filtering by bond-network uncertainty keeps molecules that are 85% valid vs 55% for random selection. This shows the uncertainty signal exists — it just gets erased by the discretization and correction steps in normal operation.

## Why 3D molecular generation is the right next step
In 3D generation, molecules are represented by atom coordinates (x, y, z) — continuous numbers that are never discretized. A slightly misplaced atom gives a slightly worse molecule (higher strain energy), not a binary valid/invalid outcome. This removes the fundamental bottleneck: uncertainty maps smoothly to quality, exactly as in the image setting where the method works. The practical payoff is in drug design, where filtering out geometrically strained molecules before expensive docking simulations could save significant compute.
