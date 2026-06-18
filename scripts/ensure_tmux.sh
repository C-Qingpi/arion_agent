#!/usr/bin/env bash
# Install tmux into ~/.local/bin (micromamba on macOS without admin; apt on Linux).
set -euo pipefail

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$PATH"

if command -v tmux >/dev/null 2>&1; then
  echo "tmux already installed: $(tmux -V)"
  exit 0
fi

link_tmux() {
  local bin="$1"
  mkdir -p "$HOME/.local/bin"
  ln -sf "$bin" "$HOME/.local/bin/tmux"
  export PATH="$HOME/.local/bin:$PATH"
}

micromamba_arch() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"
  case "$os-$arch" in
    Darwin-arm64) echo "osx-arm64" ;;
    Darwin-x86_64) echo "osx-64" ;;
    Linux-aarch64|Linux-arm64) echo "linux-aarch64" ;;
    Linux-x86_64) echo "linux-64" ;;
    *) return 1 ;;
  esac
}

install_micromamba_tmux() {
  local plat root mm
  plat="$(micromamba_arch)" || { echo "Unsupported arch: $(uname -s) $(uname -m)"; exit 1; }
  root="$HOME/.local/micromamba"
  mm="$root/bin/micromamba"
  echo "Installing tmux via micromamba ($plat) into $root ..."
  mkdir -p "$root/bin"
  if [[ ! -x "$mm" ]]; then
    curl -fsSL "https://micro.mamba.pm/api/micromamba/${plat}/latest" | tar -xj -C "$root" bin/micromamba
  fi
  MAMBA_ROOT_PREFIX="$root" "$mm" create -y -p "$root/envs/tmux" -c conda-forge tmux
  link_tmux "$root/envs/tmux/bin/tmux"
}

install_tmux() {
  case "$(uname -s)" in
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        brew install tmux
      else
        install_micromamba_tmux
      fi
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y tmux
      else
        install_micromamba_tmux
      fi
      ;;
    *)
      echo "Use WSL/Linux or macOS"
      exit 1
      ;;
  esac
}

install_tmux
command -v tmux >/dev/null
echo "tmux ready: $(tmux -V)"
