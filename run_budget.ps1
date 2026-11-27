# Award Signal - stage 1: rising areas in DoD budget requests
# Author: Tov Binyamin Brog
#
# Start it by double-clicking run_budget.bat after run_phase1.bat has set up
# Python once. It downloads each year's R-1 and P-1 from the Comptroller site,
# flags rising technology areas with the rules in config.yaml and areas.yaml,
# and writes output\budget\areas_by_year.csv and lines_flagged.csv.

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

Log "Award Signal budget run started"
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

$b = Step "Read DoD budget requests and flag rising areas" $vpy @("-m", "award_signal", "budget")
$out = Join-Path $PSScriptRoot "output\budget\areas_by_year.csv"
if ($b -eq 0 -and (Test-Path $out)) { Log "Finished. Results in $out" } else { Log "The budget step did not finish. See the newest file in the logs folder." }
Read-Host "Press Enter to close"
