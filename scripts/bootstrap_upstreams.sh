#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
vendor_root="$repo_root/vendor"
mkdir -p "$vendor_root"

clone_at() {
  local name=$1
  local url=$2
  local commit=$3
  local target="$vendor_root/$name"
  if [[ ! -d "$target/.git" ]]; then
    git clone --filter=blob:none "$url" "$target"
  fi
  git -C "$target" fetch origin "$commit" --depth 1
  git -C "$target" checkout --detach "$commit"
}

clone_at relay-opd https://github.com/zju-real/Relay-OPD.git eab21451f99e1a40fbb244f556de766d153c88f5
clone_at trd https://github.com/louieworth/trd.git 5f3894d776cb2b762a44e09f8ce8293a762e21af
clone_at ta-opd https://github.com/wyy-code/TA-OPD.git ccdf21d2066466f3d616f63cd867cc49119c45e6

echo "Pinned upstream repositories are available under $vendor_root"
