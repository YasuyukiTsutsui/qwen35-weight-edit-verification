# Keep Windows from going to sleep on idle while a long GPU queue is running. It changes NO power setting: the request
# lives only as long as this process, and ends when the watched process exits (or after -MaxHours).
#   powershell -NoProfile -File scripts/keep_awake.ps1 -WatchCommandLine queue_resume.sh
# Background: on 2026-11-04 the PC slept (Kernel-Power event 42, reason "System Idle") at 13:37 and stalled an evaluation for 6.5 h.
# The display may still turn off (only ES_SYSTEM_REQUIRED is requested). Closing the lid / manual sleep / shutdown are not prevented.
param(
    [string]$WatchCommandLine = "queue_resume.sh",
    [double]$MaxHours = 24
)
Add-Type -Namespace WSP -Name Power -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
$deadline = (Get-Date).AddHours($MaxHours)
function Watched { @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'bash|python' -and $_.CommandLine -match [regex]::Escape($WatchCommandLine) -and $_.ProcessId -ne $PID }).Count -gt 0 }
$null = [WSP.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
"[$(Get-Date -Format s)] keep-awake on; watching '$WatchCommandLine' (max $MaxHours h)"
try {
    while ((Get-Date) -lt $deadline -and (Watched)) {
        $null = [WSP.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)  # refresh
        Start-Sleep -Seconds 30
    }
} finally {
    $null = [WSP.Power]::SetThreadExecutionState($ES_CONTINUOUS)  # release
    "[$(Get-Date -Format s)] keep-awake off"
}
