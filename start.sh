#!/bin/sh
# Bootstrap and launch the AI Serving Platform (phase 1: TTS + ASR).
#
#   ./start.sh [--host H] [--port P] [--preload MODEL]... [--hf-token TOKEN]
#              [--https] [--skip-install] [--upgrade] [-- extra uvicorn args]
set -e

cd "$(dirname "$0")"
ROOT=$(pwd)

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PRELOAD="${PRELOAD_MODELS:-}"
SKIP_INSTALL=0
UPGRADE=0
HTTPS="${HTTPS:-0}"

usage() {
    cat <<EOF
Usage: ./start.sh [options] [-- extra uvicorn args]

  --host HOST         Bind address (default: 0.0.0.0, env HOST)
  --port PORT         Port (default: 8000, env PORT)
  --preload MODEL     Load a model at startup; repeatable. "defaults" loads the
                      default TTS and ASR models (env PRELOAD_MODELS, comma separated)
  --hf-token TOKEN    Hugging Face token for gated / private repositories (env HF_TOKEN)
  --https             Serve over HTTPS with a self-signed certificate (env HTTPS=1).
                      Needed to record from the microphone on other devices:
                      browsers only allow it on https:// or http://localhost
  --skip-install      Do not check or install Python dependencies
  --upgrade           Upgrade dependencies to the newest allowed versions
  -h, --help          Show this help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --host)         HOST="$2"; shift 2 ;;
        --host=*)       HOST="${1#*=}"; shift ;;
        --port)         PORT="$2"; shift 2 ;;
        --port=*)       PORT="${1#*=}"; shift ;;
        --preload)      PRELOAD="${PRELOAD:+$PRELOAD,}$2"; shift 2 ;;
        --preload=*)    PRELOAD="${PRELOAD:+$PRELOAD,}${1#*=}"; shift ;;
        --hf-token)     HF_TOKEN="$2"; export HF_TOKEN; shift 2 ;;
        --hf-token=*)   HF_TOKEN="${1#*=}"; export HF_TOKEN; shift ;;
        --https)        HTTPS=1; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --upgrade)      UPGRADE=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        --)             shift; break ;;
        *)              break ;;
    esac
done

# 1. Virtual environment -------------------------------------------------------
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_CMD=python3.12
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD=python3
    else
        echo "Error: Python 3 is not installed or not on PATH." >&2
        exit 1
    fi
    echo "Creating virtual environment with $($PYTHON_CMD --version)..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# 2. Activate ------------------------------------------------------------------
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

# 3. Dependencies (re-installed only when requirements.txt changes) ------------
if [ "$SKIP_INSTALL" -eq 0 ] && [ -f requirements.txt ]; then
    STAMP="$VENV_DIR/.requirements.cksum"
    CURRENT=$(cksum requirements.txt)
    if [ "$UPGRADE" -eq 1 ]; then
        echo "Upgrading dependencies..."
        python -m pip install --upgrade -r requirements.txt
        echo "$CURRENT" > "$STAMP"
    elif [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$CURRENT" ]; then
        echo "Installing dependencies from requirements.txt..."
        python -m pip install -r requirements.txt
        echo "$CURRENT" > "$STAMP"
    else
        echo "Dependencies up to date (use --upgrade to refresh)."
    fi
fi

# 4. Project layout ------------------------------------------------------------
mkdir -p config data/cache data/generated data/generated/images data/transcriptions data/uploads static
if [ ! -f config/models.json ]; then
    python -c "from model_manager import ModelManager; ModelManager('config/models.json')" \
        && echo "Seeded config/models.json with the default model registry."
fi

# 5. Hardware + banner ---------------------------------------------------------
OS=$(uname -s)
ARCH=$(uname -m)
CHIP="$OS $ARCH"
MEMORY=""
ACCEL="CPU"
if [ "$OS" = "Darwin" ]; then
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "$CHIP")
    BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    MEMORY="$((BYTES / 1073741824)) GB unified memory"
    if [ "$ARCH" = "arm64" ]; then
        ACCEL="MLX + PyTorch MPS"
    fi
fi
command -v ffmpeg >/dev/null 2>&1 || echo "Note: ffmpeg not found. M4A/WEBM uploads, MP3 streaming and pitch-preserving speed need it (brew install ffmpeg)."

# LAN addresses, used for the banner and the certificate.
LAN_IPS=$( (ifconfig 2>/dev/null || ip -4 addr 2>/dev/null) \
    | awk '$1 == "inet" { sub(/\/.*/, "", $2); sub(/^addr:/, "", $2); if ($2 !~ /^127\./) print $2 }' | sort -u)

# Self-signed certificate (regenerated when the machine's names/addresses change).
SCHEME="http"
if [ "$HTTPS" = "1" ]; then
    command -v openssl >/dev/null 2>&1 || { echo "Error: --https needs openssl on PATH." >&2; exit 1; }
    CERT_DIR="data/certs"
    mkdir -p "$CERT_DIR"
    SAN="DNS:localhost,IP:127.0.0.1,IP:::1"
    HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname)
    [ -n "$HOSTNAME_SHORT" ] && SAN="$SAN,DNS:$HOSTNAME_SHORT,DNS:$HOSTNAME_SHORT.local"
    for ip in $LAN_IPS; do SAN="$SAN,IP:$ip"; done
    case "$HOST" in
        0.0.0.0|127.0.0.1|localhost|::|::1) ;;
        *[!0-9.]*) SAN="$SAN,DNS:$HOST" ;;
        *) case ",$SAN," in *",IP:$HOST,"*) ;; *) SAN="$SAN,IP:$HOST" ;; esac ;;
    esac
    if [ ! -f "$CERT_DIR/cert.pem" ] || [ ! -f "$CERT_DIR/key.pem" ] \
        || [ "$(cat "$CERT_DIR/san.txt" 2>/dev/null)" != "$SAN" ] \
        || ! openssl x509 -checkend 86400 -noout -in "$CERT_DIR/cert.pem" >/dev/null 2>&1; then
        echo "Generating a self-signed certificate in $CERT_DIR/ ..."
        cat > "$CERT_DIR/openssl.cnf" <<EOF
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = AI Serving Platform (local)
[v3]
subjectAltName = $SAN
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
EOF
        openssl req -x509 -newkey rsa:2048 -nodes -days 397 \
            -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
            -config "$CERT_DIR/openssl.cnf" >/dev/null 2>&1 \
            || { echo "Error: could not generate the certificate with openssl." >&2; exit 1; }
        chmod 600 "$CERT_DIR/key.pem"
        printf '%s' "$SAN" > "$CERT_DIR/san.txt"
    fi
    SCHEME="https"
    set -- --ssl-keyfile "$CERT_DIR/key.pem" --ssl-certfile "$CERT_DIR/cert.pem" "$@"
fi

SHOW_HOST="$HOST"
[ "$HOST" = "0.0.0.0" ] && SHOW_HOST="localhost"
BASE="$SCHEME://$SHOW_HOST:$PORT"
NETWORK=""
if [ "$HOST" = "0.0.0.0" ]; then
    for ip in $LAN_IPS; do NETWORK="${NETWORK:+$NETWORK  }$SCHEME://$ip:$PORT/"; done
fi

cat <<EOF
================================================================
  AI Serving Platform - Phase 1: Text-to-Speech + Speech-to-Text
================================================================
  Hardware     $CHIP${MEMORY:+, $MEMORY}
  Acceleration $ACCEL
  Preload      ${PRELOAD:-none (models load on first request)}

  Dashboard    $BASE/
  API docs     $BASE/docs${NETWORK:+
  Network      $NETWORK}

  POST $BASE/v1/audio/speech
  POST $BASE/v1/audio/transcriptions
  POST $BASE/v1/audio/translations
  POST $BASE/v1/chat/completions      (text translation)
  POST $BASE/v1/translations
  GET  $BASE/v1/models
  GET  $BASE/v1/system/status
================================================================
EOF
if [ "$HTTPS" = "1" ]; then
    echo "HTTPS uses a self-signed certificate: accept the browser warning once per device."
    echo "API clients need verification off for it (curl -k, httpx verify=False)."
elif [ -n "$NETWORK" ]; then
    echo "Note: microphone recording from other devices needs HTTPS. Restart with --https to enable it."
fi

# 6. Launch --------------------------------------------------------------------
PRELOAD_MODELS="$PRELOAD"
export PRELOAD_MODELS HOST PORT
exec uvicorn server:app --host "$HOST" --port "$PORT" "$@"
