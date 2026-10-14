# ModelTrace 1.0.0

ModelTrace 1.0.0 is the first versioned release of the single-user local LLM auditing product.

It includes endpoint profiles, trusted API/local references, frozen audit manifests, request budgets, randomized-rank evaluation, matched power calibration, evidence exploration, and JSON/CSV/HTML reports. The React workbench, CLI, Windows/POSIX setup and Docker recipe are included.

The original branching-M identity is now consistent across the app mark, favicon, wordmark and README. .env loading, SVG delivery, Windows server cleanup and browser timing defects were corrected during release verification.

All four release jobs passed: real frontend and connected browser, native Windows, Docker and actual local HuggingFace generation/scoring. Dated evidence is in VALIDATION_REPORT.md. Follow START_HERE.md to run the product.

This release tests behavior under stated configurations. It does not prove hidden weight identity or certify detection accuracy on real hosted LLMs. Public multi-user hosting is outside the local product's deployment scope.
