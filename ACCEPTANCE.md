# ModelTrace 1.0 acceptance

ModelTrace is a single-user local auditing product. Its product scope is the workbench, CLI, compatible API adapter, local reference adapter, reproducible audits, evidence, and reports.

The first GitHub release CI run passed the real TypeScript/Vite build, connected browser workflow (28 tests passed, 1 optional checkpoint check skipped), Docker build, container health, production assets, and bounded demo. See VALIDATION_REPORT.md for the dated run and exact scope.

The updated release pipeline additionally runs native Windows setup and a real local HuggingFace checkpoint test. Their results will be recorded after the branded release commit runs. Public hosting and real-provider detection accuracy are separate from software release acceptance.
