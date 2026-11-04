<#
.SYNOPSIS
  MRP Daily Health Check — Detection → Decision → Recovery → Validation →
  Escalation → Audit, in one run. This is what Task Scheduler calls every
  morning (see Register-DailyHealthCheck-Task.ps1).

.DESCRIPTION
  App-level checks (HTTP health, SQL Server DWH, SQLite, disk/mem/cpu, log
  scanning) come from healthcheck.py. Windows-native checks (process/port
  state, Scheduled Tasks, IIS) run here. Every CRITICAL check with a defined,
  safe remediation gets one remediation attempt, then a re-check; anything
  still broken (or that has no safe automated fix) is escalated by email and
  written to the daily report either way.

.PARAMETER DryRun
  Detect, decide, and report exactly as normal, but never actually change
  anything (no process kills, no service restarts, no email sent) — logs
  "[DRYRUN] would have ..." instead. Use this to test the whole pipeline
  safely, including against the live box.

.NOTES
  Drop a file named DISABLE_AUTOHEAL in this folder to turn off remediation
  only (detection, reporting and alerting keep running) — a kill switch that
  doesn't require touching this script or Task Scheduler.
#>
param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunId = (Get-Date).ToString("yyyyMMdd-HHmmss")
$RunStart = Get-Date

# ── config / env ──────────────────────────────────────────────────
$Cfg = Get-Content (Join-Path $Here "config.json") -Raw | ConvertFrom-Json

function Import-DotEnv([string]$Path) {
    $map = @{}
    if (Test-Path $Path) {
        foreach ($line in Get-Content $Path) {
            if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
            if ($line -match '^\s*([^=]+?)\s*=\s*(.*)\s*$') {
                $map[$matches[1]] = $matches[2]
            }
        }
    }
    return $map
}
$DotEnv = Import-DotEnv (Join-Path $Here ".env")

$StateDir  = Join-Path $Here "state"
$ReportDir = Join-Path $Here "reports"
$AlertDir  = Join-Path $Here "alerts"
foreach ($d in @($StateDir, $ReportDir, $AlertDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}
$AuditFile = Join-Path $StateDir "audit.jsonl"
$RemedCountFile = Join-Path $StateDir "remediation_counts.json"

$AutoHealEnabled = -not (Test-Path (Join-Path $Here "DISABLE_AUTOHEAL"))

# ── audit trail (append-only JSONL — one line per event, never rewritten) ──
function Write-Audit([string]$Type, [hashtable]$Fields) {
    $record = [ordered]@{
        ts     = (Get-Date).ToString("o")
        run_id = $RunId
        type   = $Type
    }
    foreach ($k in $Fields.Keys) { $record[$k] = $Fields[$k] }
    ($record | ConvertTo-Json -Compress -Depth 6) | Out-File -FilePath $AuditFile -Append -Encoding utf8
}

Write-Host "=== MRP Daily Health Check — run $RunId ===" -ForegroundColor Cyan
if ($DryRun) { Write-Host "[DRYRUN] no changes will be made" -ForegroundColor Yellow }
if (-not $AutoHealEnabled) { Write-Host "[AUTOHEAL DISABLED] DISABLE_AUTOHEAL file present — detection/reporting only" -ForegroundColor Yellow }
Write-Audit "run_start" @{ dry_run = [bool]$DryRun; autoheal_enabled = $AutoHealEnabled }

# ════════════════════════════════════════════════════════════════════
# DETECTION LAYER
# ════════════════════════════════════════════════════════════════════

function Invoke-PyHealthCheck([string]$Only) {
    $py = "C:\Users\EINA-ADM\AppData\Local\Programs\Python\Python314\python.exe"
    $args = @((Join-Path $Here "healthcheck.py"))
    if ($Only) { $args += @("--only", $Only) }
    $prevEnc = $env:PYTHONIOENCODING
    $env:PYTHONIOENCODING = "utf-8"
    try {
        $raw = & $py @args 2>&1
        return ($raw | Out-String).Trim() | ConvertFrom-Json
    } finally {
        $env:PYTHONIOENCODING = $prevEnc
    }
}

function Get-ServiceProcessState($svc) {
    # The 2026-09-10 incident, generalized: more than one process bound to the
    # same port (or a large pile of half-closed sockets) is the signature of a
    # wedged service even while the process itself is technically "running".
    # @(...) around every filtered result below is deliberate, not decorative:
    # (collection | Where-Object ...).Count silently returns $null instead of
    # 1 when exactly one item matches (PowerShell only keeps .Count on 0 or 2+
    # results) — wrapping in @() forces array context so .Count is always a
    # real number, including for the single-healthy-listener case this check
    # hits on every normal run.
    $conns = @(Get-NetTCPConnection -LocalPort $svc.port -ErrorAction SilentlyContinue)
    $listeners = @($conns | Where-Object State -eq 'Listen')
    $closeWait = @($conns | Where-Object State -eq 'CloseWait').Count
    $procs = @(Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -match [regex]::Escape($svc.match) })
    [pscustomobject]@{
        ListenerCount = @($listeners | Select-Object -Unique OwningProcess).Count
        ListenerPids  = @($listeners.OwningProcess | Select-Object -Unique)
        CloseWaitCount = $closeWait
        MatchingProcessPids = @($procs.ProcessId)
    }
}

function Test-ServiceHealth($svc) {
    $state = Get-ServiceProcessState $svc
    if ($state.ListenerCount -eq 0) {
        return @{ check = $svc.name + "_process"; status = "CRITICAL"; detail = "Nothing is listening on port $($svc.port)"; data = $state }
    }
    if ($state.ListenerCount -gt 1) {
        return @{ check = $svc.name + "_process"; status = "WARN"; detail = "$($state.ListenerCount) processes bound to port $($svc.port) at once — duplicate-instance pattern (see 2026-09-10)"; data = $state }
    }
    if ($state.CloseWaitCount -gt 15) {
        return @{ check = $svc.name + "_process"; status = "WARN"; detail = "$($state.CloseWaitCount) CloseWait sockets on port $($svc.port) — early sign of a wedging service"; data = $state }
    }
    return @{ check = $svc.name + "_process"; status = "OK"; detail = "1 process listening on $($svc.port), $($state.CloseWaitCount) CloseWait"; data = $state }
}

function Test-IisHealth {
    $results = @()
    Import-Module WebAdministration -ErrorAction SilentlyContinue
    foreach ($siteName in $Cfg.iis.sites) {
        try {
            $site = Get-Website -Name $siteName -ErrorAction Stop
            $poolName = $site.applicationPool
            $poolState = (Get-WebAppPoolState -Name $poolName).Value
            if ($site.State -ne "Started" -or $poolState -ne "Started") {
                $results += @{ check = "iis_$siteName"; status = "CRITICAL"; detail = "Site state=$($site.State), pool '$poolName' state=$poolState"; data = @{ site = $siteName; pool = $poolName } }
            } else {
                $results += @{ check = "iis_$siteName"; status = "OK"; detail = "Site and pool '$poolName' both Started"; data = @{ site = $siteName; pool = $poolName } }
            }
        } catch {
            $results += @{ check = "iis_$siteName"; status = "CRITICAL"; detail = "Could not read site state: $($_.Exception.Message)"; data = @{ site = $siteName } }
        }
    }
    return $results
}

function Test-ScheduledTasksInfo {
    # Informational only (see script header) — a BootTrigger service task
    # never "completes", so LastTaskResult doesn't mean pass/fail the way it
    # would for a batch job. Real health comes from the HTTP/port checks;
    # this just keeps task state visible in the audit trail and report.
    $results = @()
    foreach ($svc in $Cfg.services) {
        try {
            $info = Get-ScheduledTaskInfo -TaskName $svc.task -ErrorAction Stop
            $results += @{ check = "task_$($svc.task)"; status = "OK"; detail = "LastRunTime=$($info.LastRunTime), LastTaskResult=$($info.LastTaskResult)"; data = @{ task = $svc.task; last_run = "$($info.LastRunTime)"; last_result = $info.LastTaskResult } }
        } catch {
            $results += @{ check = "task_$($svc.task)"; status = "WARN"; detail = "Task '$($svc.task)' not found or unreadable: $($_.Exception.Message)"; data = @{ task = $svc.task } }
        }
    }
    return $results
}

Write-Host "`n-- running checks --"
$pyOut = Invoke-PyHealthCheck
$AllChecks = @()
$AllChecks += $pyOut.results
foreach ($svc in $Cfg.services) { $AllChecks += (Test-ServiceHealth $svc) }
$AllChecks += Test-IisHealth
$AllChecks += Test-ScheduledTasksInfo
$AllChecks = $AllChecks | ForEach-Object { [pscustomobject]$_ }

foreach ($c in $AllChecks) {
    Write-Audit "check_result" @{ name = $c.check; status = $c.status; detail = $c.detail }
    $color = switch ($c.status) { "OK" {"Green"} "WARN" {"Yellow"} "CRITICAL" {"Red"} default {"Gray"} }
    Write-Host ("  [{0,-8}] {1,-28} {2}" -f $c.status, $c.check, $c.detail) -ForegroundColor $color
}

# ════════════════════════════════════════════════════════════════════
# DECISION LAYER
# ════════════════════════════════════════════════════════════════════
# Priority order: nothing else matters if the DB is down; the API is more
# upstream than the UI; resource pressure and job/task status come last.
$Priority = @("db_connectivity","service_api_process","service_api",
              "service_streamlit_process","service_streamlit",
              "service_api_proxy","iis_MrpApiProxy","memory","cpu")

function ConvertTo-PlainHashtable($Obj) {
    # ConvertFrom-Json's -AsHashtable is PowerShell 7+ only; this box runs
    # Windows PowerShell 5.1, so convert the PSCustomObject by hand (one level
    # deep is enough — every value here is read back with dot-notation, which
    # works on both hashtables and PSCustomObjects, so nested values don't
    # need converting too).
    $h = @{}
    if ($null -eq $Obj) { return $h }
    foreach ($p in $Obj.PSObject.Properties) { $h[$p.Name] = $p.Value }
    return $h
}
function Get-RemediationCounts {
    if (Test-Path $RemedCountFile) {
        return (ConvertTo-PlainHashtable (Get-Content $RemedCountFile -Raw | ConvertFrom-Json))
    }
    return @{}
}
function Save-RemediationCounts($counts) {
    $counts | ConvertTo-Json -Depth 5 | Out-File $RemedCountFile -Encoding utf8
}

function Test-CircuitBreaker([string]$CheckName, [hashtable]$Counts) {
    $today = (Get-Date).ToString("yyyy-MM-dd")
    $entry = $Counts[$CheckName]
    if (-not $entry -or $entry.date -ne $today) {
        return @{ Allowed = $true; Count = 0 }
    }
    $maxPerDay = $Cfg.circuit_breaker.max_remediations_per_check_per_day
    $cooldownMin = $Cfg.circuit_breaker.cooldown_minutes_between_attempts
    if ($entry.count -ge $maxPerDay) {
        return @{ Allowed = $false; Reason = "max $maxPerDay remediations/day already used for '$CheckName'"; Count = $entry.count }
    }
    $lastAttempt = [datetime]$entry.last_attempt
    if (((Get-Date) - $lastAttempt).TotalMinutes -lt $cooldownMin) {
        return @{ Allowed = $false; Reason = "cooldown active ($cooldownMin min) since last attempt on '$CheckName'"; Count = $entry.count }
    }
    return @{ Allowed = $true; Count = $entry.count }
}

function Update-RemediationCount([hashtable]$Counts, [string]$CheckName) {
    $today = (Get-Date).ToString("yyyy-MM-dd")
    $entry = $Counts[$CheckName]
    if (-not $entry -or $entry.date -ne $today) {
        $Counts[$CheckName] = @{ date = $today; count = 1; last_attempt = (Get-Date).ToString("o") }
    } else {
        $Counts[$CheckName] = @{ date = $today; count = ($entry.count + 1); last_attempt = (Get-Date).ToString("o") }
    }
}

# ════════════════════════════════════════════════════════════════════
# RECOVERY LAYER — one function per remediable failure type
# ════════════════════════════════════════════════════════════════════

function Restart-MrpService($svc) {
    $actions = @()
    $state = Get-ServiceProcessState $svc
    $pidsToKill = @($state.MatchingProcessPids)
    if ($pidsToKill.Count -gt 0) {
        foreach ($killPid in $pidsToKill) {
            if ($DryRun) { $actions += "[DRYRUN] would stop PID $killPid" }
            else {
                try { Stop-Process -Id $killPid -Force -ErrorAction Stop; $actions += "stopped stray/hung PID $killPid" }
                catch { $actions += "failed to stop PID $killPid : $($_.Exception.Message)" }
            }
        }
        Start-Sleep -Seconds 2
    } else {
        $actions += "no existing process found to stop"
    }
    if ($DryRun) { $actions += "[DRYRUN] would run: Start-ScheduledTask -TaskName '$($svc.task)'" }
    else {
        try { Start-ScheduledTask -TaskName $svc.task -ErrorAction Stop; $actions += "started scheduled task '$($svc.task)'" }
        catch { $actions += "FAILED to start scheduled task '$($svc.task)': $($_.Exception.Message)" }
    }
    Start-Sleep -Seconds 6   # let the process bind its port before validation re-checks
    return $actions
}

function Restart-IisTarget([string]$SiteName) {
    $actions = @()
    Import-Module WebAdministration -ErrorAction SilentlyContinue
    try {
        $site = Get-Website -Name $SiteName -ErrorAction Stop
        $poolName = $site.applicationPool
        if ($DryRun) { $actions += "[DRYRUN] would recycle app pool '$poolName' and (re)start site '$SiteName'" }
        else {
            Restart-WebAppPool -Name $poolName -ErrorAction SilentlyContinue
            Start-Website -Name $SiteName -ErrorAction SilentlyContinue
            $actions += "recycled app pool '$poolName', (re)started site '$SiteName'"
        }
    } catch { $actions += "FAILED to restart IIS site '$SiteName': $($_.Exception.Message)" }
    Start-Sleep -Seconds 3
    return $actions
}

function Repair-DbConnectivity {
    $actions = @()
    $server = $DotEnv["DB_SERVER"]
    if (-not $server) { $actions += "DB_SERVER not found in .env — cannot run network sub-check"; return $actions }
    $host_ = $server.Split('\')[0]
    if ($DryRun) { $actions += "[DRYRUN] would run: Test-NetConnection $host_ -Port 1433"; return $actions }
    try {
        $t = Test-NetConnection -ComputerName $host_ -Port 1433 -WarningAction SilentlyContinue
        if ($t.TcpTestSucceeded) { $actions += "network path to $host_`:1433 is reachable — likely a transient DB-side issue, not a network outage" }
        else { $actions += "network path to $host_`:1433 is NOT reachable — this looks infrastructure-side, restarting app services would not help" }
    } catch { $actions += "network test itself failed: $($_.Exception.Message)" }
    return $actions
}

function Invoke-ResourceRemediation([string]$CheckName, $TopProcesses) {
    $actions = @()
    $managedNames = @("streamlit","uvicorn")
    $offender = $TopProcesses | Select-Object -First 1
    if (-not $offender) { $actions += "no single dominant process identified — likely aggregate load, not one runaway process"; return $actions }
    $isManaged = $false
    foreach ($svc in $Cfg.services) {
        try {
            $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($offender.pid)" -ErrorAction Stop
            if ($p.CommandLine -match [regex]::Escape($svc.match)) {
                $isManaged = $true
                $actions += "top consumer PID $($offender.pid) ($($offender.name), $($offender.cpu_pct)% CPU) is our own '$($svc.name)' — restarting it via its scheduled task"
                $actions += (Restart-MrpService $svc)
            }
        } catch { }
    }
    if (-not $isManaged) {
        $actions += "top consumer PID $($offender.pid) ($($offender.name), $($offender.cpu_pct)% CPU) is NOT a service this monitor manages — not killing an unrecognized process automatically. Recommend manual review."
    }
    return $actions
}

# ════════════════════════════════════════════════════════════════════
# DECISION + RECOVERY + VALIDATION, driven together per failing check
# ════════════════════════════════════════════════════════════════════

$Unresolved = @()
$Resolved = @()
$NoAutoFix = @()
$RemedCounts = Get-RemediationCounts

function Get-Check($name) { return ($AllChecks | Where-Object { $_.check -eq $name } | Select-Object -First 1) }

$criticalChecks = $AllChecks | Where-Object { $_.status -eq "CRITICAL" }
$orderedCritical = @()
foreach ($p in $Priority) { $orderedCritical += ($criticalChecks | Where-Object { $_.check -eq $p }) }
$orderedCritical += ($criticalChecks | Where-Object { $Priority -notcontains $_.check })  # anything not in the priority list, run last

foreach ($check in $orderedCritical) {
    Write-Host "`n-- handling CRITICAL: $($check.check) --" -ForegroundColor Red

    if (-not $AutoHealEnabled) {
        $Unresolved += @{ check = $check.check; detail = $check.detail; reason = "auto-heal disabled (DISABLE_AUTOHEAL present)"; actions = @() }
        continue
    }

    $breaker = Test-CircuitBreaker $check.check $RemedCounts
    if (-not $breaker.Allowed) {
        Write-Host "  circuit breaker: $($breaker.Reason)" -ForegroundColor Yellow
        Write-Audit "circuit_breaker_blocked" @{ check = $check.check; reason = $breaker.Reason }
        $Unresolved += @{ check = $check.check; detail = $check.detail; reason = $breaker.Reason; actions = @() }
        continue
    }

    $actions = @()
    $hasRemediation = $true
    switch -Regex ($check.check) {
        "^service_api_process$|^service_api$"       { $actions = Restart-MrpService ($Cfg.services | Where-Object name -eq "service_api") }
        "^service_streamlit_process$|^service_streamlit$" { $actions = Restart-MrpService ($Cfg.services | Where-Object name -eq "service_streamlit") }
        "^service_api_proxy$"                        { $actions = Restart-IisTarget "MrpApiProxy" }
        "^iis_(.+)$"                                  { $actions = Restart-IisTarget $matches[1] }
        "^db_connectivity$"                           { $actions = Repair-DbConnectivity }
        "^memory$|^cpu$"                              { $actions = Invoke-ResourceRemediation $check.check $check.data.top_processes }
        default                                        { $hasRemediation = $false }
    }

    if (-not $hasRemediation) {
        $NoAutoFix += @{ check = $check.check; detail = $check.detail; reason = "no safe automated remediation defined for this failure type" }
        Write-Host "  no automated remediation defined — will report/escalate" -ForegroundColor Yellow
        continue
    }

    Update-RemediationCount $RemedCounts $check.check
    Save-RemediationCounts $RemedCounts
    Write-Audit "remediation_attempt" @{ check = $check.check; actions = $actions; dry_run = [bool]$DryRun }
    foreach ($a in $actions) { Write-Host "  -> $a" }

    # ── VALIDATION: re-check with backoff ──
    $validated = $false
    if (-not $DryRun) {
        foreach ($delay in $Cfg.retry.backoff_sec) {
            Start-Sleep -Seconds $delay
            $recheckName = $check.check -replace "_process$", ""
            $recheck = $null
            if ($check.check -match "^iis_(.+)$") { $recheck = (Test-IisHealth | Where-Object check -eq $check.check) }
            elseif ($check.check -match "_process$") { $svcDef = $Cfg.services | Where-Object { $check.check -like "$($_.name)*" }; $recheck = Test-ServiceHealth $svcDef }
            else { $py = Invoke-PyHealthCheck $recheckName; $recheck = $py.results | Select-Object -First 1 }
            Write-Host "  validating '$($check.check)' -> $($recheck.status): $($recheck.detail)"
            Write-Audit "validation" @{ check = $check.check; status = $recheck.status; detail = $recheck.detail }
            if ($recheck.status -eq "OK") { $validated = $true; break }
        }
    } else {
        $actions += "[DRYRUN] validation skipped"
    }

    if ($validated) {
        Write-Host "  RESOLVED" -ForegroundColor Green
        $Resolved += @{ check = $check.check; detail = $check.detail; actions = $actions }
    } else {
        Write-Host "  still failing after remediation + retries" -ForegroundColor Red
        $Unresolved += @{ check = $check.check; detail = $check.detail; reason = "remediation attempted but validation still failing"; actions = $actions }
    }
}

# Non-critical (WARN) checks with a defined "informational escalation" — log-file
# CRITICAL patterns are surfaced even though we don't restart anything for them,
# since they're often the root cause behind a service check that still looks OK.
$logIssues = $AllChecks | Where-Object { $_.check -like "log_*" -and $_.status -ne "OK" }

# ════════════════════════════════════════════════════════════════════
# ESCALATION LAYER
# ════════════════════════════════════════════════════════════════════

$NeedsEscalation = ($Unresolved.Count -gt 0) -or ($NoAutoFix.Count -gt 0) -or (@($logIssues | Where-Object status -eq "CRITICAL").Count -gt 0)

function Format-AlertHtml {
    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append("<h2>MRP Daily Health Check — action needed</h2>")
    [void]$sb.Append("<p>Run <code>$RunId</code> at $($RunStart.ToString('yyyy-MM-dd HH:mm')) found issue(s) that automatic recovery could not fully resolve.</p>")

    if ($Unresolved.Count -gt 0) {
        [void]$sb.Append("<h3>Unresolved after remediation attempts</h3><ul>")
        foreach ($u in $Unresolved) {
            [void]$sb.Append("<li><b>$($u.check)</b> — $($u.detail)<br><i>reason:</i> $($u.reason)<br><i>actions attempted:</i> $(($u.actions -join '; '))</li>")
        }
        [void]$sb.Append("</ul>")
    }
    if ($NoAutoFix.Count -gt 0) {
        [void]$sb.Append("<h3>No safe automated remediation exists</h3><ul>")
        foreach ($n in $NoAutoFix) {
            [void]$sb.Append("<li><b>$($n.check)</b> — $($n.detail)<br><i>$($n.reason)</i></li>")
        }
        [void]$sb.Append("</ul>")
    }
    if (@($logIssues | Where-Object status -eq "CRITICAL").Count -gt 0) {
        [void]$sb.Append("<h3>Critical log entries (root-cause indicators)</h3><ul>")
        foreach ($l in @($logIssues | Where-Object status -eq "CRITICAL")) {
            [void]$sb.Append("<li><b>$($l.check)</b> — $($l.detail)</li>")
        }
        [void]$sb.Append("</ul>")
    }

    [void]$sb.Append("<h3>Full status this run</h3><table border='1' cellpadding='4' cellspacing='0'><tr><th>Check</th><th>Status</th><th>Detail</th></tr>")
    foreach ($c in $AllChecks) {
        [void]$sb.Append("<tr><td>$($c.check)</td><td>$($c.status)</td><td>$($c.detail)</td></tr>")
    }
    [void]$sb.Append("</table>")
    [void]$sb.Append("<h3>Recommended next steps</h3><ul>")
    [void]$sb.Append("<li>RDP/console to the app server and review <code>monitoring\reports\$((Get-Date).ToString('yyyy-MM-dd')).html</code> for full detail.</li>")
    [void]$sb.Append("<li>Check <code>monitoring\state\audit.jsonl</code> for every action already attempted this run.</li>")
    [void]$sb.Append("<li>If a service keeps failing the circuit breaker, it has hit its daily auto-restart limit — investigate before re-enabling.</li>")
    [void]$sb.Append("</ul>")
    return $sb.ToString()
}

function Send-Alert([string]$Html) {
    $sent = $false
    if ($DryRun) {
        Write-Host "[DRYRUN] would send alert email / webhook now" -ForegroundColor Yellow
        Write-Audit "alert" @{ sent = $false; dry_run = $true }
        return
    }
    $smtpHost = $DotEnv["SMTP_HOST"]
    if ($smtpHost -and $smtpHost -ne "smtp.yourcompany.com" -and $DotEnv["SMTP_TO"]) {
        try {
            $msg = New-Object System.Net.Mail.MailMessage
            $msg.From = $DotEnv["SMTP_FROM"]
            foreach ($to in ($DotEnv["SMTP_TO"] -split ",")) { $msg.To.Add($to.Trim()) }
            $msg.Subject = "[MRP Monitor] Action needed — $($Unresolved.Count) unresolved, $($NoAutoFix.Count) need manual review"
            $msg.Body = $Html
            $msg.IsBodyHtml = $true
            $smtpPort = if ($DotEnv["SMTP_PORT"]) { [int]$DotEnv["SMTP_PORT"] } else { 587 }
            $client = New-Object System.Net.Mail.SmtpClient($smtpHost, $smtpPort)
            $client.EnableSsl = ($DotEnv["SMTP_USE_TLS"] -ne "false")
            if ($DotEnv["SMTP_USERNAME"]) { $client.Credentials = New-Object System.Net.NetworkCredential($DotEnv["SMTP_USERNAME"], $DotEnv["SMTP_PASSWORD"]) }
            $client.Send($msg)
            $sent = $true
            Write-Host "  alert emailed to $($DotEnv['SMTP_TO'])" -ForegroundColor Green
        } catch {
            Write-Host "  EMAIL SEND FAILED: $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        Write-Host "  SMTP not configured (monitoring\.env) — falling back to file + Event Log" -ForegroundColor Yellow
    }

    # File + Event Log fire regardless of email success — a second, independent
    # channel so "the mail server was also down" doesn't mean total silence.
    $alertFile = Join-Path $AlertDir "ALERT_$RunId.html"
    $Html | Out-File -FilePath $alertFile -Encoding utf8
    try {
        if (-not [System.Diagnostics.EventLog]::SourceExists("MRP Daily Monitor")) {
            New-EventLog -LogName Application -Source "MRP Daily Monitor" -ErrorAction SilentlyContinue
        }
        Write-EventLog -LogName Application -Source "MRP Daily Monitor" -EntryType Warning -EventId 1001 `
            -Message "MRP health check $RunId found unresolved issues. See $alertFile" -ErrorAction SilentlyContinue
    } catch { }
    Write-Audit "alert" @{ sent = $sent; file = $alertFile }
}

if ($NeedsEscalation) {
    $html = Format-AlertHtml
    Send-Alert $html
} else {
    Write-Host "`nNo escalation needed." -ForegroundColor Green
    Write-Audit "alert" @{ sent = $false; reason = "nothing to escalate" }
}

# ════════════════════════════════════════════════════════════════════
# REPORTING
# ════════════════════════════════════════════════════════════════════
$reportPath = Join-Path $ReportDir "$((Get-Date).ToString('yyyy-MM-dd')).html"
$statusRows = ($AllChecks | ForEach-Object {
    $cls = switch ($_.status) { "OK" {"ok"} "WARN" {"warn"} "CRITICAL" {"crit"} default {""} }
    "<tr class='$cls'><td>$($_.check)</td><td>$($_.status)</td><td>$($_.detail)</td></tr>"
}) -join "`n"

$actionRows = (($Resolved + $Unresolved) | ForEach-Object {
    "<tr><td>$($_.check)</td><td>$(($_.actions) -join '<br>')</td></tr>"
}) -join "`n"

$html = @"
<!DOCTYPE html><html><head><meta charset='utf-8'><title>MRP Health Report $((Get-Date).ToString('yyyy-MM-dd'))</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#1b231e;background:#f3f5f1}
h1{font-size:22px} h2{font-size:16px;margin-top:28px}
table{border-collapse:collapse;width:100%;background:#fff;margin-top:8px}
td,th{border:1px solid #d7ddd1;padding:6px 10px;text-align:left;font-size:13px}
tr.ok td:nth-child(2){color:#2d6b5e;font-weight:600}
tr.warn td:nth-child(2){color:#8f631a;font-weight:600}
tr.crit td:nth-child(2){color:#a3402f;font-weight:600}
.summary{display:flex;gap:16px;margin-top:12px}
.tile{background:#fff;border:1px solid #d7ddd1;border-radius:8px;padding:12px 18px}
.tile b{font-size:20px;display:block}
</style></head><body>
<h1>MRP Daily Health Report — $((Get-Date).ToString('yyyy-MM-dd HH:mm'))</h1>
<p>Run <code>$RunId</code> · Auto-heal: $(if ($AutoHealEnabled) {"enabled"} else {"DISABLED"}) $(if ($DryRun) {"· DRY RUN"} else {""})</p>
<div class='summary'>
  <div class='tile'><b>$(@($AllChecks | Where status -eq 'OK').Count)</b>OK</div>
  <div class='tile'><b>$(@($AllChecks | Where status -eq 'WARN').Count)</b>Warn</div>
  <div class='tile'><b>$(@($AllChecks | Where status -eq 'CRITICAL').Count)</b>Critical</div>
  <div class='tile'><b>$($Resolved.Count)</b>Auto-resolved</div>
  <div class='tile'><b>$($Unresolved.Count + $NoAutoFix.Count)</b>Need attention</div>
</div>
<h2>All checks this run</h2>
<table><tr><th>Check</th><th>Status</th><th>Detail</th></tr>$statusRows</table>
<h2>Automatic actions taken</h2>
<table><tr><th>Check</th><th>Actions</th></tr>$actionRows</table>
</body></html>
"@
$html | Out-File -FilePath $reportPath -Encoding utf8
$html | Out-File -FilePath (Join-Path $ReportDir "latest.html") -Encoding utf8
Write-Host "`nReport written: $reportPath"

$runResult = if (($Unresolved.Count -gt 0) -or ($NoAutoFix.Count -gt 0)) { "needs_attention" } else { "healthy" }
Write-Audit "run_summary" @{
    result = $runResult
    ok = @($AllChecks | Where status -eq 'OK').Count
    warn = @($AllChecks | Where status -eq 'WARN').Count
    critical = @($AllChecks | Where status -eq 'CRITICAL').Count
    resolved = $Resolved.Count
    unresolved = $Unresolved.Count
    no_auto_fix = $NoAutoFix.Count
    duration_sec = [math]::Round(((Get-Date) - $RunStart).TotalSeconds, 1)
}

Write-Host "`n=== run complete: $runResult ===" -ForegroundColor $(if ($runResult -eq "healthy") {"Green"} else {"Red"})
if ($runResult -eq "needs_attention") { exit 1 } else { exit 0 }
