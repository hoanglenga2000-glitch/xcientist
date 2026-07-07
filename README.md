# EvoMind - XCIENTIST AI Research Workstation

An auditable AI research workstation for Kaggle and MLE-Bench style machine
learning work. It combines a terminal research agent, a browser EvoMind gateway,
experiment evidence, report generation, literature/RAG support, and strict
submission gates.

Default gateway:

```text
http://127.0.0.1:8088/?page=control
```

## What This System Provides

- **Terminal Agent**: `evomind` opens a Claude-Code-like research
  terminal for task selection, planning, readiness checks, and gated execution.
- **Web Workstation**: a Next.js workstation gateway with pages for control, tasks,
  data, GPU, evidence, literature, workflow, code, runtime, experiments,
  reports, gates, and settings.
- **Four-Layer Architecture**:
  - Layer 1: Multi-Agent Research OS for task parsing, data audit, code,
    validation, report, and artifact workflow.
  - Layer 2: MLEvolve-style Search Controller for search graph, best-so-far,
    branch strategy, and self-evolution.
  - Layer 3: XCIENTIST Research Harness for validation contracts, claim audit,
    leakage checks, and evidence boundaries.
  - Layer 4: Memory / Benchmark / Kaggle Feedback for retrospective memory,
    benchmark tracking, and official-result separation.
- **Credential Safety**: API keys and SSH secrets are stored with Windows DPAPI
  helper scripts, not in the repository.
- **Human Gate**: official Kaggle submission and medal/rank claims remain
  blocked unless an explicit human approval gate and official response artifact
  exist.

## Quick Start

```powershell
git clone https://github.com/hoanglenga2000-glitch/xcientist.git EvoMind
cd EvoMind
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
```

Then start the workstation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_verified_workstation.ps1 restart
```

Open:

```text
http://127.0.0.1:8088/?page=control
```

Check the terminal agent:

```powershell
evomind ready
evomind
```

## Configuration

Use the guided CLI:

```powershell
evomind setup
```

Or install secrets with DPAPI helper scripts:

```powershell
powershell -File scripts\manage_deepseek_secret.ps1 install-key -ApiKey sk-xxx
powershell -File scripts\manage_kaggle_secret.ps1 install-token -ApiToken <token>
powershell -File scripts\install_hpc_ssh_credential_from_stdin.ps1 -User <ssh-user> -HostName <login-node> -Port <port> -RemoteWorkspace <remote-dir>
```

Notes:

- DeepSeek or Claude/Anthropic provides the LLM brain for planning, code
  generation, and audit workflows.
- Kaggle credentials are required for official downloads and submissions.
- GPU/HPC credentials are optional and only required for remote training jobs.
- Official Kaggle submission remains Human Gate controlled.

## CLI Commands

```powershell
evomind                       # Enter the EvoMind research terminal
autokaggle                    # Compatibility alias for EvoMind
evomind ready                   # Show setup and resource readiness
evomind setup                   # Configure LLM/Kaggle/compute
evomind competitions titanic    # Browse/search Kaggle competitions
evomind task add <KaggleURL>    # Register a competition task
evomind run <task>              # Start gated audited execution
evomind watch -f                # Follow event stream
evomind memory                  # Inspect retrospective memory
evomind dashboard start         # Start/open the workstation
evomind official ...            # Official Kaggle CLI passthrough
kaggle-official ...            # Direct official Kaggle CLI passthrough
```

## Verification

Run the new-user release gate:

```powershell
python scripts\verify_new_user_release_readiness.py --write-report
```

Run the full new-user release acceptance suite:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_new_user_release_acceptance.ps1
```

Run the full workstation launch gate:

```powershell
python scripts\verify_workstation_launch_readiness.py --write-report
```

Typical safe demo state:

```text
status: passed
release_state: ready_for_new_user_evomind_gateway
launch_state: demo_ready_training_blocked_by_gpu
```

This means the EvoMind gateway, terminal agent, documentation, CLI routing, and
non-training workflows are ready. It does not mean GPU training, official Kaggle
submission, rank, medal, or MLE-Bench-75 performance is proven.

## Project Layout

```text
src/xsci/                         Terminal EvoMind Research Agent
src/research_os/                  Search, memory, and agent-side research OS
web/research-agent-workstation/   Next.js workstation UI and API routes
scripts/                          Install, secret, launch, and verification tools
configs/                          Task and external resource configuration
experiments/                      Experiment records and artifacts
reports/                          Generated audit and readiness reports
workspace/                        Runtime reports, smoke outputs, and state
docs/                             User and research documentation
tests/                            Python CLI and workflow tests
```

## Release Boundary

The system is designed to be honest about evidence:

- Do not claim official Kaggle score, rank, medal, or top-30 status without a
  Kaggle response artifact.
- Do not claim MLE-Bench-75 parity from a small subset of tasks.
- Do not bypass the workstation gate with ad hoc training.
- Do not store API keys, tokens, cookies, SSH keys, or passwords in git.
- Record failed tasks and blocked gates instead of hiding them.

For detailed onboarding, see:

```text
docs/EvoMind_New_User_Final_Setup_Guide_20260707.md
docs/EvoMind_新用户最终配置使用手册_20260707.md
docs/NEW_USER_ONBOARDING_GUIDE.md
docs/RELEASE_CHECKLIST.md
```
