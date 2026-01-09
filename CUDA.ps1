$cudaRoot = Join-Path $PWD "Briefcase\v13.1"
$env:CUDA_PATH = $cudaRoot
$env:PATH = (Join-Path $cudaRoot "bin") + ";" + (Join-Path $cudaRoot "bin\x64") + ";" + $env:PATH

uv run turbulence
