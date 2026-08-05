-- CreateTable
CREATE TABLE "tasks" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "task_type" TEXT NOT NULL,
    "target" TEXT,
    "metric" TEXT,
    "status" TEXT NOT NULL,
    "priority" TEXT,
    "owner" TEXT,
    "config_path" TEXT,
    "task_dir" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "experiment_runs" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "task_id" TEXT NOT NULL,
    "output_dir" TEXT,
    "status" TEXT NOT NULL,
    "best_model" TEXT,
    "metrics_json" TEXT,
    "validation_status" TEXT,
    "process_id" INTEGER,
    "started_at" DATETIME,
    "finished_at" DATETIME,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "experiment_runs_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "tasks" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "action_logs" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "action" TEXT NOT NULL,
    "task_id" TEXT,
    "run_id" TEXT,
    "message" TEXT NOT NULL,
    "artifact_path" TEXT,
    "metadata_json" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "action_logs_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "experiment_runs" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "action_logs_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "tasks" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "workflows" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "task_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "nodes_json" TEXT NOT NULL,
    "edges_json" TEXT NOT NULL,
    "published_at" DATETIME,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "workflows_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "tasks" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "gates" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "task_id" TEXT NOT NULL,
    "run_id" TEXT,
    "gate_type" TEXT NOT NULL,
    "decision" TEXT NOT NULL,
    "reviewer" TEXT,
    "evidence_json" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "decided_at" DATETIME,
    CONSTRAINT "gates_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "experiment_runs" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "gates_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "tasks" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "evidence" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "task_id" TEXT NOT NULL,
    "run_id" TEXT,
    "label" TEXT NOT NULL,
    "artifact_path" TEXT,
    "hash" TEXT,
    "source" TEXT,
    "claim_binding" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "evidence_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "experiment_runs" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "evidence_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "tasks" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "reports" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "task_id" TEXT NOT NULL,
    "run_id" TEXT,
    "title" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "markdown_content" TEXT,
    "content_json" TEXT,
    "markdown_path" TEXT,
    "docx_path" TEXT,
    "selected_section" TEXT,
    "submitted_at" DATETIME,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL,
    CONSTRAINT "reports_run_id_fkey" FOREIGN KEY ("run_id") REFERENCES "experiment_runs" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "reports_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "tasks" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "connector_statuses" (
    "provider" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "configured" BOOLEAN NOT NULL,
    "detail" TEXT,
    "updated_at" DATETIME NOT NULL
);

-- CreateTable
CREATE TABLE "settings" (
    "key" TEXT NOT NULL PRIMARY KEY,
    "value_json" TEXT NOT NULL,
    "updated_at" DATETIME NOT NULL
);

-- CreateIndex
CREATE INDEX "experiment_runs_task_id_created_at_idx" ON "experiment_runs"("task_id", "created_at");

-- CreateIndex
CREATE INDEX "action_logs_task_id_created_at_idx" ON "action_logs"("task_id", "created_at");

-- CreateIndex
CREATE INDEX "workflows_task_id_updated_at_idx" ON "workflows"("task_id", "updated_at");

-- CreateIndex
CREATE INDEX "gates_task_id_created_at_idx" ON "gates"("task_id", "created_at");

-- CreateIndex
CREATE INDEX "evidence_task_id_created_at_idx" ON "evidence"("task_id", "created_at");

-- CreateIndex
CREATE INDEX "reports_task_id_updated_at_idx" ON "reports"("task_id", "updated_at");

