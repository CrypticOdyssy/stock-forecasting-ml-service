#!/bin/sh
set -e

host_port="$1"
shift 1

echo "Waiting for $host_port..."

# nc -z host port  → split host:port using sh parameter expansion
host="${host_port%%:*}"
port="${host_port##*:}"

while ! nc -z "$host" "$port"; do
  echo "  ."
  sleep 1
done

echo "Database at $host:$port is available — running command"
exec "$@"
