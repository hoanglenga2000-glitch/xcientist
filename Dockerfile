FROM node:20-bookworm-slim

ENV NEXT_TELEMETRY_DISABLED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKSTATION_ROOT=/app
ENV DATABASE_URL=file:/app/web/research-agent-workstation/prisma/workstation.db

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir -r /app/requirements.txt

COPY web/research-agent-workstation/package*.json /app/web/research-agent-workstation/
WORKDIR /app/web/research-agent-workstation
RUN npm install -g npm@11.6.2 \
    && npm ci

WORKDIR /app
COPY . /app

# Production-stability gate: fail the image build if any first-party Python is
# broken (syntax) or unimportable. Tests run in CI/pre-commit where dev deps are
# present; this in-image gate needs only the control-plane runtime deps.
RUN python3 scripts/run_ci_checks.py --skip-tests

WORKDIR /app/web/research-agent-workstation
RUN npx prisma generate \
    && npx prisma db push \
    && npm run build

EXPOSE 3090

HEALTHCHECK --interval=20s --timeout=8s --start-period=20s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3090/api/workstation-summary').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["sh", "-c", "npx prisma db push && npx next start --hostname 0.0.0.0 --port 3090"]
