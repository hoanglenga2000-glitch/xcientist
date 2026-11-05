# Roll the pool cutover back (config files + python package), then restart with c05.
$ErrorActionPreference='Stop'
. 'C:\ProgramData\EvoMind\bundle\scripts\lib\Runtime.ps1'
Assert-Administrator
$stg='C:\ProgramData\EvoMind\staging\pool-98456-20261105'
$b=(Get-Content "$stg\backup-dir.txt" -Raw).Trim()
$T='C:\ProgramData\EvoMind\byoa\tenants\tenant_ba0ef9d3767f2fb385b856e0'
Copy-Item "$b\gpu-pool.json" 'C:\ProgramData\EvoMind\state\gpu-pool.json' -Force
Copy-Item "$b\hpc-binding.json" "$T\hpc-binding.json" -Force
Copy-Item "$b\gpu-vram.json" 'C:\ProgramData\EvoMind\data\workspace\runtime\config\gpu-vram.json' -Force
Copy-Item "$b\node-config.json" "$script:ConfigPath.restore.tmp" -Force; Move-Item "$script:ConfigPath.restore.tmp" $script:ConfigPath -Force
$rt='C:\ProgramData\EvoMind\bundle\runtime'
$aside=(Get-Content "$stg\python3-aside-id.txt" -Raw -EA 0)
if($aside){ $aside=$aside.Trim(); if(Test-Path "$rt\evomind_runtime.$aside.stage"){ Rename-Item "$rt\evomind_runtime" ("evomind_runtime." + [guid]::NewGuid().ToString('N') + ".rolledback"); Rename-Item "$rt\evomind_runtime.$aside.stage" 'evomind_runtime'; "python restored" } }
& 'C:\ProgramData\EvoMind\bundle\scripts\Seal-Release.ps1' | Out-Null
"ROLLBACK_OK"
