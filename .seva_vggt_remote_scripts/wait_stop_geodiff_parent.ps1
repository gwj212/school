$ErrorActionPreference = 'Continue'
$logPath = 'D:\Desktop\vggt_large\.seva_vggt_remote_scripts\wait_stop_geodiff_parent.log'
for ($attempt = 1; $attempt -le 240; $attempt++) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $output = & ssh -o BatchMode=yes -o ConnectTimeout=5 -T -p 10247 root@10.44.6.60 "kill 4145634 2>/dev/null || true; echo PARENT_STOPPED; ps -p 4145767 -o pid,etime,stat --no-headers || true" 2>&1
    Add-Content -LiteralPath $logPath -Value "$stamp attempt=$attempt exit=$LASTEXITCODE $output"
    if ($LASTEXITCODE -eq 0 -and ($output -match 'PARENT_STOPPED')) {
        exit 0
    }
    Start-Sleep -Seconds 15
}
exit 1
