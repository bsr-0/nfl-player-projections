## CRITICAL: Bug Discovery and Code Integrity Rules  
This repository has a **high historical rate of bugs, regressions, incorrect assumptions, and defects caused by seemingly small code changes**. Treat the existing codebase as potentially unreliable until verified.  
These rules are mandatory for every agent and override the instinct to remain narrowly focused on the current task.  
**1. STOP IMMEDIATELY WHEN YOU FIND A BUG**  
If, while working on any objective, you discover **any definite bug, error, broken behavior, incorrect calculation, invalid assumption, stale logic, inconsistent implementation, schema mismatch, leakage risk, dead code that is still being relied upon, or other material defect**, **STOP pursuing your original objective temporarily.**  
Do not simply note the issue and continue.  
Do not defer it because it is “outside the scope” of the current task.  
Do not work around it.  
First determine whether the defect can affect:  
* the current task;  
* downstream code;  
* model inputs or outputs;  
* validation or evaluation;  
* production artifacts;  
* other strategies or paths through the system;  
* historical results or reported metrics.  
If it can affect any of these, **fix or properly resolve the defect before continuing with the original objective.**  
The correct priority is:  
**Correctness → integrity of the codebase → original task objective → optimization/refinement.**  
A task completed on top of known broken code is not considered successfully completed.  
**2. DO NOT ASSUME AN UNRELATED BUG IS ACTUALLY UNRELATED**  
This repository has repeatedly contained defects that initially appeared unrelated to the requested change but turned out to affect the correctness of the system.  
Therefore, when you encounter suspicious code, **investigate before proceeding.**  
Examples include:  
* unexpected hardcoded values;  
* duplicated logic;  
* inconsistent feature schemas;  
* different code paths implementing the same concept differently;  
* stale comments or documentation that contradict the implementation;  
* suspicious defaults;  
* unexplained special cases;  
* dead or apparently unreachable branches;  
* functions whose inputs do not match their callers;  
* calculations that appear dimensionally or statistically incorrect;  
* probability conversions or scoring logic that seem inconsistent;  
* train/serve feature mismatches;  
* data leakage or temporal-causality concerns;  
* stale generated artifacts;  
* frontend/backend schema mismatches;  
* code that silently falls back to zeros, defaults, or missing values;  
* code that appears to work only because of a particular current dataset;  
* tests that validate implementation details rather than actual correctness.  
If something looks wrong, **pause and verify it rather than rationalizing it away.**  
**3. SCAN SURROUNDING CODE BEFORE WRITING NEW CODE**  
Before adding or substantially modifying code, inspect the relevant:  
* callers;  
* callees;  
* data structures;  
* schemas;  
* related functions;  
* neighboring modules;  
* tests;  
* configuration;  
* generated artifacts;  
* downstream consumers.  
Do not assume that the function or file you were explicitly asked to modify contains the entire relevant implementation.  
A small change can expose or create defects elsewhere.  
When introducing a new function, feature, calculation, model component, or data transformation, determine:  
1. What supplies its inputs?  
2. What assumptions does it make?  
3. What consumes its outputs?  
4. Are those assumptions actually true?  
5. Are there parallel implementations that must remain consistent?  
6. Could the change invalidate existing tests or downstream behavior?  
7. Could the change introduce a new train/serve, temporal, schema, or data-contract mismatch?  
**4. NEVER PATCH AROUND A BUG JUST TO FINISH THE TASK**  
Do not add:  
* arbitrary guards;  
* silent fallbacks;  
* hardcoded corrections;  
* special-case branches;  
* randomization;  
* magic constants;  
* duplicated implementations;  
* try/except blocks that hide failures;  
* test-specific behavior;  
* compatibility hacks;  
merely to make the current task pass.  
If the underlying logic is wrong, **fix the underlying logic.**  
If you cannot safely fix it, stop and clearly document the blocker rather than disguising it.  
**5. NEW CODE REQUIRES EXTRA SKEPTICISM**  
Assume that every new line of code has some probability of introducing a regression.  
Before considering new code complete, check:  
* input/output types and shapes;  
* null and missing-value behavior;  
* edge cases;  
* empty inputs;  
* duplicate inputs;  
* ordering assumptions;  
* indexing;  
* joins and keys;  
* train/test or historical/future boundaries;  
* feature availability at prediction time;  
* probability normalization;  
* numerical stability;  
* deterministic behavior where required;  
* consistency with existing implementations;  
* downstream consumers;  
* tests covering the actual behavior.  
Prefer the **smallest correct change** over a clever abstraction or broad refactor unless a broader change is required for correctness.  
**6. WHEN A DISCOVERY CHANGES THE SCOPE, PAUSE**  
If investigation reveals that the original task is based on a false assumption, **pause the original plan.**  
Do not blindly continue executing a plan that has been invalidated by new evidence.  
Instead:  
1. State what was discovered.  
2. Determine the impact.  
3. Correct the defect or revise the plan.  
4. Re-check affected code.  
5. Only then resume the original objective.  
The goal is not to maximize the number of requested tasks completed per session.  
The goal is to leave the repository **more correct than you found it**.  
**7. VERIFY, DO NOT TRUST**  
Historical code in this repository should not be treated as authoritative merely because:  
* it already exists;  
* it has comments explaining it;  
* a previous agent wrote it;  
* a test passes;  
* a metric looks reasonable;  
* it has been used in production;  
* the implementation appears mathematically plausible.  
When correctness matters, **measure or trace the behavior.**  
If a claim can be verified directly from the code, data, or execution, verify it rather than relying on assumptions.  
**8. FINAL CHECK BEFORE DECLARING SUCCESS**  
Before reporting a task as complete:  
* Re-read the changes.  
* Inspect the surrounding code.  
* Run relevant tests.  
* Check for regressions.  
* Check that new behavior matches the actual data/schema contracts.  
* Check that no discovered bug was merely deferred.  
* Check that generated artifacts are consistent with source code.  
* Check that the implementation does not introduce a second, subtly different version of existing logic.  
**Do not declare success while knowingly leaving a material correctness defect unresolved.**  
**Core Principle**  
**If you see a bug, stop. Investigate it. Fix it or explicitly establish that it is harmless. Then continue.**  
This repository has a history of bugs propagating through otherwise reasonable-looking code. **Do not optimize for task completion at the expense of codebase correctness.**  
