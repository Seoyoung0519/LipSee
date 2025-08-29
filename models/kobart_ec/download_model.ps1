# models/kobart_ec/download_model.ps1
$Tag   = "ec-kobart-v0.1"
$Asset = "model.safetensors"
$Url   = "https://github.com/Seoyoung0519/LipSee/releases/download/$Tag/$Asset"
$Dst   = "models\kobart_ec\$Asset"
$Sha   = "142056d820f4ac177e0aee662a54cb600d9503474b1e07dbcf09ca7de17225d0"

New-Item -ItemType Directory -Force -Path (Split-Path $Dst) | Out-Null

# 이미 있으면 해시 검증만 하고 종료
if (Test-Path $Dst) {
  $hash = (Get-FileHash $Dst -Algorithm SHA256).Hash.ToLower()
  if ($hash -eq $Sha.ToLower()) {
    Write-Host "[OK] model exists & hash verified: $Dst"
    exit 0
  } else {
    Write-Host "[WARN] Hash mismatch. Re-downloading..."
    Remove-Item $Dst -Force
  }
}

Invoke-WebRequest -Uri $Url -OutFile $Dst

$hash = (Get-FileHash $Dst -Algorithm SHA256).Hash.ToLower()
if ($hash -ne $Sha.ToLower()) { throw "SHA256 mismatch: $hash" }
Write-Host "[DONE] Downloaded and verified $Asset -> $Dst"
