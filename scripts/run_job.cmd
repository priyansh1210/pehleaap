@echo off
rem Run one pehleaap module on this machine (used on the GPU box by remote.ps1).
rem   scripts\run_job.cmd train.train_eot --config configs\train_eot.yaml
rem Pulls the latest commit first so the GPU box never runs stale code.
cd /d "%~dp0.."
git pull --ff-only || exit /b 1
if not exist runs mkdir runs
.venv\Scripts\python.exe -m pehleaap.%* > "runs\last_%1.log" 2>&1
