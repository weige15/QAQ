# QAQ worker-session and recovery rules

A worker process, terminal, pane, or window is temporary execution state. The
feature branch, worktree, commits, and files are the durable work. Losing a
session is not a captain decision and must not create a permission prompt.

## Automatic recovery

An authorized current-objective operation already permits relaunching or replacing
a worker session when read-only checks establish that:

- the recorded worktree still exists;
- its physical path and Git top-level are unchanged;
- the expected feature branch and preserved `HEAD` are present;
- no other live write-enabled worker owns that worktree;
- no unexplained Git operation or out-of-scope change is present; and
- recovery will not reset, clean, stash, rebase, overwrite, replace, or delete
  preserved work.

When those checks pass, reload the current controller or FirstMate brief,
repeat `.pi/rules/qaq-runtime.md`, bind the replacement worker to the same
worktree, and continue the same objective immediately. Report recovery in the next
normal material update rather than asking first.

A clean relaunch, another test process, a replaced pane, reloading controller
instructions, or retrying a command after a transient process failure is
routine.

## Blocker boundary

Escalate when safe identity cannot be established, another writer may be live,
preserved work would need to be discarded or rewritten, credentials or external
material are missing, or the same substantive obstacle has been reached twice
without new evidence.

State the exact failed check, what was preserved, and the smallest material
decision or external action needed. Do not escalate merely because a window is
missing, the destination advanced, a test needs rerunning, or a new shell needs
its environment activated.
