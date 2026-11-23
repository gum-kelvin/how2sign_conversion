# How2Sign evaluation status

## Purpose

This repository converts How2Sign metadata and frontal RGB video into the manifests and feature layout required by ViPo-SLT. The research objective is to run the complete How2Sign corpus through conversion, preprocessing, training, and evaluation, then compare it with a documented PHOENIX2014-T baseline. This repository does not include raw data, features, model weights, or experiment outputs.

## What the converter implements

- Read-only inspection of source metadata and media.
- Portable manifest and configuration generation outside the ViPo-SLT checkout.
- Resumable CLIP spatial and VideoMAE motion extraction from the frontal RGB stream.
- Validation of identifiers, splits, targets, numeric arrays, widths, and feature coverage.
- A reversible ViPo-SLT integration patch that enables manifest-driven How2Sign loading.

How2Sign currently uses frontal RGB only. Pose conversion is deliberately rejected because it has not been validated. CTC is deliberately disabled because the converter does not manufacture gloss labels.

## Verified evidence

The companion [audit](../AUDIT.md) records the offline acceptance work. It reports successful compilation, a 73-test suite with one dependency-path skip, static contract checks, and a synthetic loader smoke test against the patched ViPo-SLT loader. This evidence establishes integration behavior only; it does not establish real-data extraction, GPU compatibility, full-corpus coverage, or translation quality.

### Test execution context

The suite deliberately covers two ViPo-SLT states. Loader, guard, and smoke tests require a checkout with the integration patch applied. Installer apply/revert tests require a clean, unpatched checkout. Running the entire suite against only one of those states produces expected fixture failures; those failures are not evidence that the converter logic changed. Record the target repository revision and patch state with every test run.

## Dataset inventory on 23 November 2026

A strict read-only inspection was run on a local How2Sign snapshot. The snapshot contained 1,976 files and 38,664,008,851 bytes. Its annotations contained 31,165 train rows, 1,741 validation rows, and 2,357 test rows.

The inspection failed with 31,166 errors:

- 31,165 train rows lacked an available frontal source video.
- The test annotation content was tab-delimited despite the `.csv` filename, so a CSV parser saw one column instead of the required metadata columns.
- 132 frontal validation videos were available and resolved 1,741 validation samples; no train or test frontal source videos were available.

This snapshot must not proceed to full feature extraction or training. The required remediation is to acquire or mount the train and test frontal media, parse or normalize the test metadata correctly, and rerun strict inspection with zero unresolved media or metadata errors.

## Evidence required before performance claims

1. Record source release, access terms, license, metadata hashes, media inventory, and split counts.
2. Require a strict inspection report with no unresolved source-media or metadata failures.
3. Run a small real GPU extraction pilot and record model versions, feature widths, timings, storage, errors, and retries.
4. Validate the pilot output strictly before full-corpus extraction.
5. Run a real one-batch GPU training smoke test with finite loss, finite gradients, and an optimizer step.
6. Preserve code commit, configuration hash, manifest fingerprint, environment, seed, predictions, metrics, costs, and failures for each full How2Sign and PHOENIX run.
7. Compare outcomes only after documenting all material differences in data, preprocessing, weights, hyperparameters, hardware, and evaluation protocol.

## Publication and deployment boundary

No model should be described as validated for a live sign-language translation application until real-data extraction, training, evaluation, checkpoint selection, inference latency, failure behavior, and privacy/safety requirements have all been recorded and reviewed.
