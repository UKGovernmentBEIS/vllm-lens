---
name: scientific-debug
description: Use this skill whenever the user asks to debug something, fix a bug, troubleshoot an issue, or mentions "scientific debugging". Any debugging request should use this skill.
---

# Scientific Debugging

Stop. Do not try random fixes. Your goal is to **understand the root cause** before changing anything.

## Step 1: Read the evidence

Read the error, logs, and relevant code. If the cause is genuinely obvious from what you can see — state specifically what you believe is wrong, why you're confident, fix it, and you're done.

But be honest with yourself: RL training has made you overconfident. You often *feel* sure but are wrong, leading to a cycle of random patches that bloat the codebase. If you're not >95% certain of the root cause, do NOT attempt a fix. Go to Step 2.

## Step 2: Get more information

You need more logging. This is the most important step — almost any bug becomes obvious with enough visibility.

- Increase log verbosity: set debug log levels, add env vars that enable verbose output, import loggers and set their level.
- If you're unsure how to get more logging for the specific library/framework involved, **search the internet** for how to enable debug logging for that tool. Don't guess from potentially stale knowledge.
- Add targeted print/log statements around the suspicious area.
- If it's not a Slurm/remote job, use breakpoints and a debugger.

Run the code again with enhanced logging.

## Step 3: Reassess

With the new logs, are you now >95% confident in the root cause?

- **Yes**: State the cause specifically, explain why you're confident, and make the fix.
- **No**: Go back to Step 2. Dig deeper — read the source code of underlying libraries, find what env vars or config options control logging, check if you can set the logger's level programmatically. You can always get enough information to understand what's happening. Do not skip this and guess.

Repeat Steps 2-3 until you truly understand the problem. Only then fix it.

## Step 4: Verify the fix

After applying your fix, rerun the code to confirm the bug is actually gone. If the job is expensive (large Slurm jobs, long GPU runs, etc.), ask the user before rerunning.
