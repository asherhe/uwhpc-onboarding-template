param(
  [int]$n_runs = 20
)

# rebuild benchmark executable
cmake --build --preset benchmark

$run_data = @()

for ($i = 1; $i -le $n_runs; $i++) {
  $json = .\build\benchmark\uwhpc_benchmark.exe
  Write-Output $json
  # parse to powershell object
  $data = $json | ConvertFrom-Json
  $run_data += $data
}

$avg = ($run_data | Measure-Object -Property "score" -Average).Average
Write-Output "Average score: $avg"
