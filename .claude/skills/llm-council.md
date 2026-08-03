---
name: llm-council
description: "Run any question, idea, or decision through a council of 5 AI advisors who independently analyze it, peer-review each other anonymously, and synthesize a final verdict. Based on Karpathy's LLM Council methodology. MANDATORY TRIGGERS: 'council this', 'run the council', 'war room this', 'pressure-test this', 'stress-test this', 'debate this'. STRONG TRIGGERS (use when combined with a real decision or tradeoff): 'should I X or Y', 'which option', 'what would you do', 'is this the right move', 'validate this', 'get multiple perspectives', 'I can't decide', 'I'm torn between'. Do NOT trigger on simple yes/no questions, factual lookups, or casual 'should I' without a meaningful tradeoff (e.g. 'should I use markdown' is not a council question). DO trigger when the user presents a genuine decision with stakes, multiple options, and context that suggests they want it pressure-tested from multiple angles."
---

# LLM Council

You ask one AI a question, you get one answer. That answer might be great. It might be mid. You have no way to tell because you only saw one perspective.

The council fixes this. It runs your question through 5 independent advisors, each thinking from a fundamentally different angle. Then they review each other's work. Then a chairman synthesizes everything into a final recommendation that tells you where the advisors agree, where they clash, and what you should actually do.

Before any of that happens, a **Point Person** (Step 0) checks whether the question has already been answered in a prior council, is a status check on outstanding action items, or is a follow-up about a past session. Most repeat questions never need a full council re-run. See the `council-point-person` skill.

This is adapted from Andrej Karpathy's LLM Council. He dispatches queries to multiple models, has them peer-review each other anonymously, then a chairman produces the final answer. We do the same thing inside Claude using sub-agents with different thinking lenses instead of different models.

---

## When to Run the Council

The council is for questions where being wrong is expensive.

Good council questions:
- "Should I launch a $97 workshop or a $497 course?"
- "Which of these 3 positioning angles is strongest?"
- "I'm thinking of pivoting from X to Y. Am I crazy?"
- "Here's my landing page copy. What's weak?"
- "Should I hire a VA or build an automation first?"

Bad council questions:
- "What's the capital of France?" (one right answer, no need for perspectives)
- "Write me a tweet" (creation task, not a decision)
- "Summarize this article" (processing task, not judgment)

The council shines when there's genuine uncertainty and the cost of a bad call is high. If you already know the answer and just want validation, the council will likely tell you things you don't want to hear. That's the point.

---

## The Five Advisors

Each advisor thinks from a different angle. They're not job titles or personas. They're thinking styles that naturally create tension with each other.

### 1. The Contrarian
Actively looks for what's wrong, what's missing, what will fail. Assumes the idea has a fatal flaw and tries to find it. If everything looks solid, digs deeper. The Contrarian is not a pessimist. They're the friend who saves you from a bad deal by asking the questions you're avoiding.

### 2. The First Principles Thinker
Ignores the surface-level question and asks "what are we actually trying to solve here?" Strips away assumptions. Rebuilds the problem from the ground up. Sometimes the most valuable council output is the First Principles Thinker saying "you're asking the wrong question entirely."

### 3. The Statistician
Treats every claim as a hypothesis and asks what the data actually supports. Separates signal from noise, anecdote from evidence, and correlation from causation. Asks the uncomfortable questions: What's the sample size? What's the base rate? What's the selection bias? What's the effect size versus the variance? Is this a pattern, or three points on a chart we're calling a trend? The Statistician is not a pedant — they're the advisor who stops you from betting the farm on a signal that was always noise. Where the Contrarian hunts for flaws and the First Principles Thinker rethinks the framing, the Statistician audits the evidence itself. If a decision rests on numbers, the Statistician interrogates those numbers. If it rests on intuition, the Statistician demands the cheapest experiment that would confirm or kill the intuition before real capital gets committed.

### 4. The Outsider
Has zero context about you, your field, or your history. Responds purely to what's in front of them. This is the most underrated advisor. Experts develop blind spots. The Outsider catches the curse of knowledge: things that are obvious to you but confusing to everyone else.

### 5. The Executor
Only cares about one thing: can this actually be done, and what's the fastest path to doing it? Ignores theory, strategy, and big-picture thinking. The Executor looks at every idea through the lens of "OK but what do you do Monday morning?" If an idea sounds brilliant but has no clear first step, the Executor will say so.

**Why these five:** They create three natural tensions. Contrarian vs Executor (what could go wrong vs what can we ship). First Principles vs Statistician (rebuild the frame vs trust only what the evidence supports). The Outsider sits in the middle keeping everyone honest by seeing what fresh eyes see. Together they cover the four failure modes of bad decisions: hidden downside (Contrarian), wrong question (First Principles), weak evidence (Statistician), insider blind spots (Outsider), and no path to execution (Executor).

---

## How a Council Session Works

### Step 0: Triage via the Point Person

Before convening any advisors, invoke the `council-point-person` skill. It scans recent council transcripts and classifies the user's question into one of five buckets:

- **Bucket 1 — Already Answered.** Return the prior verdict. Do not convene.
- **Bucket 2 — Follow-up on a Prior Session.** Point Person answers from the transcript. Do not convene.
- **Bucket 3 — Status Check.** Point Person returns outstanding action items. Do not convene.
- **Bucket 4 — Re-council with Changed Conditions.** Point Person summarizes the delta and, on user confirmation, hands off a tightened framed question and a reduced advisor roster (typically 2 advisors). Convene a *partial* council from Step 1.
- **Bucket 5 — Genuinely New.** Proceed to Step 1 with the full 5-advisor council.

If the user explicitly says "fresh council," "skip triage," or "new council on this," skip Step 0 and go straight to Step 1. Otherwise, Step 0 is mandatory — it's the primary mechanism preventing redundant $9-subagent re-runs of settled questions.

When Step 0 short-circuits the session (Buckets 1, 2, or 3), do NOT write a new `council-transcript-*.md`. The Point Person's output is conversational; no transcript artifact is produced for non-council outcomes.

When Step 0 hands off a partial council (Bucket 4), use the Point Person's reduced advisor list for Step 2 and its tightened framed question as the input to all remaining steps. Peer review in Step 3 drops to 2 reviewers if only 2 advisors responded. The Chairman template is unchanged.

### Step 1: Frame the Question (with context enrichment)

When the user says "council this" (or any trigger phrase), do two things before framing:

**A. Scan the workspace for context.** The user's question is often just the tip of the iceberg. Their Claude setup likely contains files that would dramatically improve the council's output. Before framing, quickly scan for and read any relevant context files:

- `CLAUDE.md` or `claude.md` in the project root or workspace (business context, preferences, constraints)
- Any `memory/` folder (audience profiles, voice docs, business details, past decisions)
- Any files the user explicitly referenced or attached
- Recent council transcripts in this folder (to avoid re-counciling the same ground)
- Any other context files that seem relevant to the specific question (e.g., if they're asking about pricing, look for revenue data, past launch results, audience research)

Use `Glob` and quick `Read` calls to find these. Don't spend more than 30 seconds on this. You're looking for the 2-3 files that would give advisors the context they need to give specific, grounded advice instead of generic takes.

**B. Frame the question.** Take the user's raw question AND the enriched context and reframe it as a clear, neutral prompt that all five advisors will receive. The framed question should include:

1. The core decision or question
2. Key context from the user's message
3. Key context from workspace files (business stage, audience, constraints, past results, relevant numbers)
4. What's at stake (why this decision matters)

Don't add your own opinion. Don't steer it. But DO make sure each advisor has enough context to give a specific, grounded answer rather than generic advice.

If the question is too vague ("council this: my business"), ask one clarifying question. Just one. Then proceed.

Save the framed question for the transcript.

### Step 2: Convene the Council (5 sub-agents in parallel)

Spawn all 5 advisors simultaneously as sub-agents. Each gets:

1. Their advisor identity and thinking style (from the descriptions above)
2. The framed question
3. A clear instruction: respond independently. Do not hedge. Do not try to be balanced. Lean fully into your assigned perspective. If you see a fatal flaw, say it. If you see massive upside, say it. Your job is to represent your angle as strongly as possible. The synthesis comes later.

Each advisor should produce a response of 150-300 words. Long enough to be substantive, short enough to be scannable.

**Sub-agent prompt template:**
```
You are [Advisor Name] on an LLM Council.

Your thinking style: [advisor description from above]

A user has brought this question to the council:

---
[framed question]
---

Respond from your perspective. Be direct and specific. Don't hedge or try to be balanced. Lean fully into your assigned angle. The other advisors will cover the angles you're not covering.

Keep your response between 150-300 words. No preamble. Go straight into your analysis.
```

### Step 3: Peer Review (3 sub-agents in parallel)

This is the step that makes the council more than just "ask 5 times." It's the core of Karpathy's insight — but we don't need 5 reviewers to get it. Three independent peer reviews cover the same ground with 40% less compute and no meaningful loss of signal.

Collect all 5 advisor responses. Anonymize them as Response A through E (randomize which advisor maps to which letter so there's no positional bias).

Spawn 3 peer review sub-agents in parallel. Each reviewer sees all 5 anonymized responses and answers three questions:

1. Which response is the strongest and why? (pick one)
2. Which response has the biggest blind spot and what is it?
3. What did ALL responses miss that the council should consider?

Pick the 3 reviewers by rotating through the advisor roster (e.g., Contrarian, Statistician, Outsider for one session; First Principles, Executor, Contrarian for the next). The goal is reviewer diversity, not reviewer quantity.

**Reviewer prompt template:**
```
You are reviewing the outputs of an LLM Council. Five advisors independently answered this question:

---
[framed question]
---

Here are their anonymized responses:

**Response A:** [response]
**Response B:** [response]
**Response C:** [response]
**Response D:** [response]
**Response E:** [response]

Answer these three questions. Be specific. Reference responses by letter.

1. Which response is the strongest? Why?
2. Which response has the biggest blind spot? What is it missing?
3. What did ALL five responses miss that the council should consider?

Keep your review under 150 words. Be direct. No preamble.
```

### Step 4: Chairman Synthesis

This is the final step. One agent gets everything: the original question, all 5 advisor responses (now de-anonymized so you can see which advisor said what), and the 3 peer reviews.

The chairman's job is to turn raw advisor output into something the user can actually act on before closing the tab. The old "here's what everyone said" summary is dead weight — if the user wanted five opinions they'd have read five opinions. What they want is a verdict and a sequenced path forward.

The chairman produces output in this order, optimized for top-down scanning:

1. **Bottom Line** -- one sentence: what to do. One sentence: what not to do. That's it. The user who only reads this section should already know what's next.

2. **Critical Next Steps** -- an ordered, sequenced list of 3 to 5 actions. Not a bulleted wishlist. Each step must unlock the next, and the FIRST step is marked `**DO TODAY:**`. Every step names a single owner-level action (verb + object), a rough time cost (hours/days), and a success signal (how you'll know that step worked). If a step can't be defined this concretely, it doesn't belong on the list.

3. **Council Convergence** -- the points multiple advisors landed on independently. High-confidence signals. Two or three bullets, not a recap.

4. **Council Disagreement** -- the real tradeoff at the center of the decision. Name the axis of disagreement in a single phrase (e.g., "speed vs. rigor"), present both sides in two sentences each, and state which side the chairman is taking and why.

5. **Blind Spots Caught in Review** -- things only the peer review round surfaced. Skip this section if there were none — don't manufacture them.

6. **Kill Criteria** -- the signal that would tell the user to stop, reverse course, or re-council. One or two concrete tripwires. This is what separates a real plan from optimistic momentum.

**Chairman prompt template:**
```
You are the Chairman of an LLM Council. Your job is to turn 5 advisor responses and 3 peer reviews into a verdict the user can act on immediately.

The question brought to the council:
---
[framed question]
---

ADVISOR RESPONSES:

**The Contrarian:** [response]
**The First Principles Thinker:** [response]
**The Statistician:** [response]
**The Outsider:** [response]
**The Executor:** [response]

PEER REVIEWS:
[the 3 peer reviews]

Produce the verdict using this exact structure and order. Be direct. No hedging, no "it depends," no recaps of what advisors said. The user wants clarity and a path, not a summary.

## Bottom Line
**Do:** [one sentence.]
**Don't:** [one sentence.]

## Critical Next Steps
1. **DO TODAY:** [verb + object]. Time: [hours/days]. Success signal: [observable outcome].
2. [verb + object]. Time: [...]. Success signal: [...].
3. [verb + object]. Time: [...]. Success signal: [...].
[Up to 5 total. Each step must unlock the next. No wishlists.]

## Council Convergence
- [high-confidence signal]
- [high-confidence signal]

## Council Disagreement
**The real tradeoff:** [one phrase — e.g., "speed vs. rigor"].
**Side A:** [two sentences.]
**Side B:** [two sentences.]
**Chairman's call:** [which side, and the single reason why.]

## Blind Spots Caught in Review
[Only include if the peer review round surfaced something genuinely new. Otherwise omit this section entirely.]

## Kill Criteria
- [concrete tripwire that would tell the user to stop or reverse course]
- [optional second tripwire]

End of output. Do not add closing commentary.
```

### Step 5: Save the Full Transcript

Save the complete council session as `council-transcript-[timestamp].md`. Markdown is the only output format — no HTML, no duplicated artifact. The chairman's verdict is structured for top-down scanning in any markdown viewer (editor, GitHub, terminal), so a separate rendered report adds cost without adding signal.

The transcript must be organized in this exact order so the most actionable content is at the top:

1. **Header** -- date and one-line question.
2. **Chairman's Verdict** -- the full output from Step 4. This is the section the user actually reads; it must come first.
3. **Original Question** and **Framed Question** -- for reference.
4. **Advisor Responses** -- all 5, de-anonymized, in the order listed in Section "The Five Advisors".
5. **Peer Reviews** -- the 3 reviews, with the anonymization mapping (A→which advisor) revealed at the top of this section.

Use `##` for top-level sections and `###` for advisor/reviewer subsections. Keep the file to a single `.md` file — no companion HTML, no embedded images, no JSON sidecar.

Print the file path back to the user when done. Do not open it automatically.

---

## Output Format

Every council session produces exactly one file:

```
council-transcript-[timestamp].md
```

Markdown only. No HTML reports. The chairman's verdict sits at the top of the transcript so it's the first thing the user reads; the advisor responses and peer reviews live below for anyone who wants to dig in.

---

## Important Notes

- **Step 0 is mandatory unless the user opts out.** The Point Person prevents redundant council runs on questions already answered. Skip it only when the user explicitly says "fresh council" or similar.
- **Always spawn all 5 advisors in parallel** (or 2 for a Bucket 4 partial council). Sequential spawning wastes time and lets earlier responses bleed into later ones.
- **Always spawn the 3 peer reviewers in parallel** once advisor responses are in. Same reasoning. Drop to 2 reviewers for a Bucket 4 partial council.
- **Always anonymize for peer review.** If reviewers know which advisor said what, they'll defer to certain thinking styles instead of evaluating on merit.
- **The chairman can disagree with the majority.** If 4 out of 5 advisors say "do it" but the reasoning of the 1 dissenter is strongest, the chairman should side with the dissenter and explain why — and say so explicitly in the **Chairman's call** line.
- **Don't council trivial questions.** If the user asks something with one right answer, just answer it. The council is for genuine uncertainty where multiple perspectives add value.
- **Markdown-only output.** The chairman's verdict is structured to be scannable in raw markdown. Do not generate HTML, PDF, or any other rendered artifact. One file per session: `council-transcript-[timestamp].md`.
- **Don't pad the verdict.** If there are no genuine blind spots from peer review, omit that section. If the council didn't really disagree, say so in one line instead of inventing a tradeoff. Sections that get manufactured for completeness dilute the real signal.
