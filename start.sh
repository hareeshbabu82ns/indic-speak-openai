#!/bin/sh
# Bootstrap and launch the AI Serving Platform (phase 1: TTS + ASR).
#
#   ./start.sh [run] [options] [-- extra uvicorn args]   run in the foreground
#   ./start.sh start [options] [-- extra uvicorn args]   run in the background
#   ./start.sh stop | status | restart [options] | logs [-f] [-n N]
#
# options: [--host H] [--port P] [--preload MODEL]... [--hf-token TOKEN]
#          [--https] [--skip-install] [--upgrade]
set -e

cd "$(dirname "$0")"
ROOT=$(pwd)

# Background service state (data/ is git-ignored).
RUN_DIR="data/run"
PID_FILE="$RUN_DIR/server.pid"
STATE_FILE="$RUN_DIR/server.state"
ARGS_FILE="$RUN_DIR/server.args"
LOG_DIR="data/logs"
LOG_FILE="$LOG_DIR/server.log"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

# Shell-quote one argument so a list can be saved and restored with eval.
quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# PID of the background server, if it is running (guards against PID reuse).
server_pid() {
    [ -f "$PID_FILE" ] || return 1
    _pid=$(cat "$PID_FILE" 2>/dev/null)
    [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null || return 1
    ps -p "$_pid" -o command= 2>/dev/null | grep -q "uvicorn" || return 1
    echo "$_pid"
}

load_state() {
    S_SCHEME=http; S_HOST=0.0.0.0; S_PORT=8000; S_STARTED=""
    # shellcheck disable=SC1090
    [ -f "$STATE_FILE" ] && . "./$STATE_FILE"
    S_PROBE="$S_HOST"
    case "$S_HOST" in 0.0.0.0|::|"") S_PROBE=127.0.0.1 ;; esac
    S_URL="$S_SCHEME://$S_PROBE:$S_PORT"
}

health_json() { curl -sf -k --max-time "${2:-3}" "$1/health" 2>/dev/null; }

cmd_status() {
    load_state
    if ! PID=$(server_pid); then
        echo "Server: stopped"
        [ -f "$PID_FILE" ] && echo "  (stale pid file $PID_FILE removed)" && rm -f "$PID_FILE"
        return 3
    fi
    UPTIME=$(ps -p "$PID" -o etime= 2>/dev/null | tr -d ' ')
    RSS_KB=$(ps -p "$PID" -o rss= 2>/dev/null | tr -d ' ')
    echo "Server: running (pid $PID, up ${UPTIME:-?}, RSS $(( ${RSS_KB:-0} / 1024 )) MB)"
    echo "  URL      $S_SCHEME://$( [ "$S_PROBE" = 127.0.0.1 ] && echo localhost || echo "$S_PROBE" ):$S_PORT/"
    echo "  Started  ${S_STARTED:-?}"
    echo "  Log      $LOG_FILE"
    if JSON=$(health_json "$S_URL"); then
        printf '%s' "$JSON" | python3 -c '
import json, sys
h = json.load(sys.stdin)
print("  Health   %s (version %s, device %s)" % (h.get("status"), h.get("version"), h.get("device")))
print("  Loaded   %s" % (", ".join(h.get("loaded_models") or []) or "none"))
d = h.get("defaults") or {}
if d: print("  Defaults %s" % ", ".join("%s=%s" % kv for kv in sorted(d.items())))
' 2>/dev/null || echo "  Health   ok"
    else
        echo "  Health   not responding yet (starting up or busy; see ./start.sh logs)"
        return 1
    fi
}

cmd_stop() {
    if ! PID=$(server_pid); then
        echo "Server is not running."
        rm -f "$PID_FILE"
        return 0
    fi
    printf 'Stopping server (pid %s)' "$PID"
    kill -TERM "$PID" 2>/dev/null || true
    i=0
    while kill -0 "$PID" 2>/dev/null; do
        i=$((i + 1))
        if [ "$i" -gt "$STOP_TIMEOUT" ]; then
            printf ' still running after %ss, sending SIGKILL' "$STOP_TIMEOUT"
            kill -KILL "$PID" 2>/dev/null || true
            break
        fi
        printf '.'
        sleep 1
    done
    echo " stopped."
    rm -f "$PID_FILE"
}

cmd_logs() {
    [ -f "$LOG_FILE" ] || { echo "No log file yet ($LOG_FILE)."; return 1; }
    FOLLOW=0; LINES=100
    while [ $# -gt 0 ]; do
        case "$1" in
            -f|--follow) FOLLOW=1; shift ;;
            -n|--lines)  LINES="$2"; shift 2 ;;
            -n*)         LINES="${1#-n}"; shift ;;
            *) echo "Unknown logs option: $1" >&2; return 2 ;;
        esac
    done
    if [ "$FOLLOW" -eq 1 ]; then exec tail -n "$LINES" -f "$LOG_FILE"; fi
    tail -n "$LINES" "$LOG_FILE"
}

# Command dispatch --------------------------------------------------------------
DAEMON=0
case "${1:-}" in
    run)     shift ;;
    start)   shift; DAEMON=1 ;;
    stop)    shift; cmd_stop; exit $? ;;
    status)  shift; set +e; cmd_status; exit $? ;;
    logs|log) shift; cmd_logs "$@"; exit $? ;;
    restart)
        shift
        if [ $# -eq 0 ] && [ -f "$ARGS_FILE" ]; then
            eval "set -- $(cat "$ARGS_FILE")"
            echo "Restarting with the previous options: $*"
        fi
        cmd_stop
        exec "$0" start "$@"
        ;;
esac

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PRELOAD="${PRELOAD_MODELS:-}"
SKIP_INSTALL=0
UPGRADE=0
HTTPS="${HTTPS:-0}"

usage() {
    cat <<EOF
Usage: ./start.sh [command] [options] [-- extra uvicorn args]

Commands:
  run                 Run in the foreground (default when no command is given)
  start               Run in the background; output goes to $LOG_FILE
  stop                Stop the background server (SIGTERM, SIGKILL after ${STOP_TIMEOUT}s)
  restart [options]   Stop, then start again (reuses the last start options when none are given)
  status              Show PID, uptime, memory, URL, health and loaded models
  logs [-f] [-n N]    Show the last N (default 100) log lines; -f follows

Options (run / start / restart):
  -d, --daemon        Same as the "start" command

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
        -d|--daemon|--background) DAEMON=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        --)             shift; break ;;
        *)              break ;;
    esac
done

# Extra uvicorn args as given (before --https adds its own), for restart.
EXTRA_ARGS=""
for a in "$@"; do EXTRA_ARGS="$EXTRA_ARGS $(quote "$a")"; done

if PID=$(server_pid); then
    echo "Error: the server is already running in the background (pid $PID)." >&2
    echo "Use ./start.sh status, ./start.sh restart or ./start.sh stop." >&2
    exit 1
fi

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

if [ "$DAEMON" -ne 1 ]; then
    exec uvicorn server:app --host "$HOST" --port "$PORT" "$@"
fi

# Background mode: detach uvicorn, record its PID, options and address.
mkdir -p "$RUN_DIR" "$LOG_DIR"
if [ -f "$LOG_FILE" ] && [ "$(wc -c < "$LOG_FILE")" -gt 10485760 ]; then
    mv -f "$LOG_FILE" "$LOG_FILE.1"            # keep one 10 MB generation
fi
SAVED="--host $(quote "$HOST") --port $(quote "$PORT")"
[ -n "$PRELOAD" ] && SAVED="$SAVED --preload $(quote "$PRELOAD")"
[ "$HTTPS" = "1" ] && SAVED="$SAVED --https"
[ -n "$EXTRA_ARGS" ] && SAVED="$SAVED --$EXTRA_ARGS"
printf '%s\n' "$SAVED" > "$ARGS_FILE"       # the HF token is never written to disk
printf '\n===== %s  starting on %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$BASE" >> "$LOG_FILE"
PYTHONUNBUFFERED=1 nohup uvicorn server:app --host "$HOST" --port "$PORT" "$@" >> "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"
cat > "$STATE_FILE" <<EOF
S_SCHEME=$SCHEME
S_HOST=$HOST
S_PORT=$PORT
S_STARTED='$(date '+%Y-%m-%d %H:%M:%S')'
EOF

PROBE="$HOST"
case "$HOST" in 0.0.0.0|::) PROBE=127.0.0.1 ;; esac
printf 'Started in the background (pid %s). Waiting for it to respond' "$PID"
i=0
while [ "$i" -lt "${START_TIMEOUT:-60}" ]; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo " failed."
        echo "The server exited during startup. Last log lines ($LOG_FILE):" >&2
        tail -n 30 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    if health_json "$SCHEME://$PROBE:$PORT" 2 >/dev/null; then
        echo " ready."
        echo "Manage it with: ./start.sh status | logs -f | restart | stop"
        exit 0
    fi
    printf '.'
    sleep 1
    i=$((i + 1))
done
echo
echo "Still starting (preloading models can take a while). Check ./start.sh status or ./start.sh logs -f."
