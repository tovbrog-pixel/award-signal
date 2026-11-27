# Award Signal - Phase 1 run on DoD daily announcements (war.gov)
# Author: Tov Binyamin Brog
#
# Start it by double-clicking run_announcements.bat after run_phase1.bat has
# set up Python once. It reads every DoD daily contract announcement since 2015
# (about an hour the first time, cached after that), matches the companies and
# writes the report to output\announcements\report.html.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
# A new log file each run: OneDrive can lock a file it is syncing, which blocks appends
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "logs") | Out-Null
$log = Join-Path $PSScriptRoot ("logs\run_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line -ForegroundColor Cyan
    Add-Content -Path $log -Value $line
}

function Step([string]$title, [string]$exe, [string[]]$arguments) {
    Log "START $title"
    & $exe @arguments 2>&1 | ForEach-Object {
        $text = "$_"
        Write-Host $text
        Add-Content -Path $log -Value $text
    }
    $code = $LASTEXITCODE
    Log "END $title (exit code $code)"
    return $code
}

function Find-Python {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $exe = (& py -3 -c "import sys; print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
    }
    return $null
}

Log "Award Signal announcement run started"
$vpy = Join-Path $env:LOCALAPPDATA "award-signal\venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Log "Run run_phase1.bat first to set up Python."; Read-Host "Press Enter to close"; exit 1 }
Step "Install packages" $vpy @("-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", "requirements.txt") | Out-Null
$t = Step "Tests" $vpy @("-m", "pytest", "-q", "-p", "no:cacheprovider")
if ($t -ne 0) { Log "Tests failed; stopping."; Read-Host "Press Enter to close"; exit 1 }

$contactFile = Join-Path $env:LOCALAPPDATA "award-signal\sec_contact.txt"
if (-not $env:SEC_USER_AGENT) {
    if (Test-Path $contactFile) {
        $contact = (Get-Content $contactFile -Raw).Trim()
    } else {
        $contact = Read-Host "Type the email address SEC should see, then press Enter"
        Set-Content -Path $contactFile -Value $contact
    }
    $env:SEC_USER_AGENT = "Individual researcher $contact"
}

$a = Step "Read DoD announcements (about an hour the first time)" $vpy @("-m", "award_signal", "awards", "--source", "announcements")
if ($a -ne 0) { Log "Announcement download failed; stopping."; Read-Host "Press Enter to close"; exit 1 }
$m = Step "Match recipients to companies" $vpy @("-m", "award_signal", "match")
if ($m -ne 0) { Log "Matching failed; stopping."; Read-Host "Press Enter to close"; exit 1 }
$r = Step "Event study" $vpy @("-m", "award_signal", "run", "--out", "announcements")

$report = Join-Path $PSScriptRoot "output\announcements\report.html"
if ($r -eq 0 -and (Test-Path $report)) {
    Log "Finished. Opening $report"
    Start-Process $report
} else {
    Log "The event study did not finish. See the newest file in the logs folder."
}
Read-Host "Press Enter to close"
