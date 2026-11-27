# Award Signal - Phase 1 setup and run for Windows
# Author: Tov Binyamin Brog
#
# Start it by double-clicking run_phase1.bat. It installs Python 3.12 if
# needed, builds a virtual environment in %LOCALAPPDATA%\award-signal (outside
# OneDrive), runs the tests and demo, then the real award, match and run steps.
# Everything it prints also goes to a new file in the logs folder.

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

Log "Award Signal Phase 1 run started"

$py = Find-Python
if (-not $py) {
    Step "Install Python 3.12 (winget)" "winget" @("install", "-e", "--id", "Python.Python.3.12", "--scope", "user",
        "--accept-package-agreements", "--accept-source-agreements") | Out-Null
    $py = Find-Python
}
if (-not $py) {
    Log "Python could not be found or installed. See the newest file in the logs folder."
    Read-Host "Press Enter to close"
    exit 1
}
Log "Using Python: $py"

$venv = Join-Path $env:LOCALAPPDATA "award-signal\venv"
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Step "Create virtual environment" $py @("-m", "venv", $venv) | Out-Null
}
Step "Install packages" $vpy @("-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", "requirements.txt") | Out-Null
$t = Step "Tests" $vpy @("-m", "pytest", "-q", "-p", "no:cacheprovider")
if ($t -ne 0) { Log "Tests failed; stopping."; Read-Host "Press Enter to close"; exit 1 }
Step "Synthetic demo" $vpy @("-m", "award_signal", "demo") | Out-Null

# The contact typed on the first run is kept in %LOCALAPPDATA%\award-signal (outside OneDrive)
$contactFile = Join-Path $env:LOCALAPPDATA "award-signal\sec_contact.txt"
if (-not $env:SEC_USER_AGENT) {
    if (Test-Path $contactFile) {
        $contact = (Get-Content $contactFile -Raw).Trim()
    } else {
        Write-Host ""
        Write-Host "SEC asks every program that downloads its data to include a contact email." -ForegroundColor Yellow
        $contact = Read-Host "Type the email address SEC should see, then press Enter"
        Set-Content -Path $contactFile -Value $contact
    }
    $env:SEC_USER_AGENT = "Individual researcher $contact"
}

$a = Step "Download DoD awards (can take an hour or more)" $vpy @("-m", "award_signal", "awards")
if ($a -ne 0) { Log "Award download failed; stopping."; Read-Host "Press Enter to close"; exit 1 }
$m = Step "Match recipients to companies" $vpy @("-m", "award_signal", "match")
if ($m -ne 0) { Log "Matching failed; stopping."; Read-Host "Press Enter to close"; exit 1 }
$r = Step "Event study" $vpy @("-m", "award_signal", "run")

$report = Join-Path $PSScriptRoot "output\report.html"
if ($r -eq 0 -and (Test-Path $report)) {
    Log "Finished. Opening $report"
    Start-Process $report
} else {
    Log "The event study did not finish. See the newest file in the logs folder."
}
Read-Host "Press Enter to close"
