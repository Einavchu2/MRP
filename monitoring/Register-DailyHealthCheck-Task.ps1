<#
.SYNOPSIS
  One-time setup: registers the "MRP Daily Health Check" Scheduled Task and
  hardens the two existing app-launch tasks ("Data API", "Streamlit App")
  with auto-restart-on-crash settings they don't currently have.

.DESCRIPTION
  Run this once, interactively, as the account that should own the task
  (it will prompt for that account's password — Scheduled Tasks with a
  Password logon type, same as the existing "Data API"/"Streamlit App"
  tasks, need a stored credential; there is no way to script that without
  either prompting or embedding a password in a file, and the latter is
  exactly what this project does not do).

  Idempotent — re-running it updates the existing task/settings rather
  than erroring or duplicating them.

.PARAMETER Time
  Local time to run the daily check. Default 06:30 — before the work day
  starts, after any overnight batch activity.

.PARAMETER SkipHardening
  Registers the new task only; leaves "Data API" / "Streamlit App" as-is.
#>
param(
    [string]$Time = "06:30",
    [switch]$SkipHardening
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptPath = Join-Path $Here "Invoke-DailyHealthCheck.ps1"
$TaskName = "MRP Daily Health Check"

Write-Host "=== Registering '$TaskName' ===" -ForegroundColor Cyan
$cred = Get-Credential -Message "Account to run the daily health check as (needs rights to restart the app's scheduled tasks and IIS site)"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`"" `
    -WorkingDirectory $Here

$trigger = New-ScheduledTaskTrigger -Daily -At $Time

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 5) `
    -DontStopOnIdleEnd -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

$principal = New-ScheduledTaskPrincipal -UserId $cred.UserName -LogonType Password -RunLevel Highest

$task = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "Daily MRP monitoring & self-healing check — see monitoring\README.md"

Register-ScheduledTask -TaskName $TaskName -InputObject $task -User $cred.UserName -Password $cred.GetNetworkCredential().Password -Force | Out-Null
Write-Host "Registered '$TaskName' to run daily at $Time as $($cred.UserName)." -ForegroundColor Green

if (-not $SkipHardening) {
    Write-Host "`n=== Hardening existing app-launch tasks ===" -ForegroundColor Cyan
    Write-Host "Note: RestartCount/RestartInterval only fires when the task's own"
    Write-Host "process EXITS (crash) — it will not catch a process that is still"
    Write-Host "running but wedged/unresponsive. That case is what the daily health"
    Write-Host "check above exists to catch. The two layers are complementary, not"
    Write-Host "redundant."

    foreach ($existingTaskName in @("Data API", "Streamlit App")) {
        try {
            $existing = Get-ScheduledTask -TaskName $existingTaskName -ErrorAction Stop
            $hardenedSettings = New-ScheduledTaskSettingsSet `
                -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 2) `
                -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                -ExecutionTimeLimit ([TimeSpan]::Zero)   # zero = no limit, these are long-running servers
            Set-ScheduledTask -TaskName $existingTaskName -Settings $hardenedSettings | Out-Null
            Write-Host "  hardened '$existingTaskName': restart up to 3x, 2 min apart, on crash" -ForegroundColor Green
        } catch {
            Write-Host "  could not update '$existingTaskName': $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

Write-Host "`nDone. Next steps:"
Write-Host "  1. Copy monitoring\.env.example to monitoring\.env and fill in SMTP details."
Write-Host "  2. Test end to end without touching anything: powershell -File `"$ScriptPath`" -DryRun"
Write-Host "  3. Review monitoring\reports\latest.html after the first real (non-dry-run) run."
