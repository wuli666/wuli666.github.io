---
summary: Agents that work on real screens and real files, and the reinforcement learning that trains them.
---

## OpenDazi

A desktop AI agent that works with the files on your machine. I wrote nearly all of it; it is built with Electron and TypeScript, and private for now.

- **Chat** about a project, with the plan and tool calls visible, and keep steering while it works
- **Board**: every card is a task with its own conversation, materials and results. It runs in the background and comes back for review
- **Workflows**: once a task passes its checks it can be saved as a workflow, then rerun on new inputs
- **Meetings**: local or cloud transcription into a transcript, a summary and a mind map, with action items sent to the board
- **Memory**: preferences and project conventions picked up from conversations, which you can review and edit
- Scheduled tasks that run a prompt on a timetable, system audio capture on macOS, and Windows and Linux builds for internal testers

## verl-agentic

An RL framework for Android GUI agents. Private for now.

Rather than maintain a fork, it is being rebuilt as a plugin on top of upstream veRL. The plugin focuses on step-level learning signals, which is where GUI tasks are hardest:

- grouping steps by screen state, so an action is compared against others taken from the same state
- forking rollouts from environment snapshots instead of replaying from the start
- self-distillation from a teacher that sees privileged information

## langextract-vllm

A provider plugin that lets Google's [LangExtract](https://github.com/google/langextract) run structured extraction against a vLLM server.

- Source: [wuli666/langextract-vllm](https://github.com/wuli666/langextract-vllm), 30 stars
- Listed in LangExtract's community provider registry ([google/langextract#244](https://github.com/google/langextract/pull/244))
