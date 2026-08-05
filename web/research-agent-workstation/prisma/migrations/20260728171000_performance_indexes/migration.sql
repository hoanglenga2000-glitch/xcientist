-- Add global time-order indexes used by the lightweight workstation summary.
-- All statements are additive and safe to apply repeatedly through Prisma's
-- migration ledger.
CREATE INDEX "tasks_updated_at_idx" ON "tasks"("updated_at");
CREATE INDEX "tasks_created_at_idx" ON "tasks"("created_at");
CREATE INDEX "experiment_runs_created_at_idx" ON "experiment_runs"("created_at");
CREATE INDEX "action_logs_created_at_idx" ON "action_logs"("created_at");
CREATE INDEX "workflows_updated_at_idx" ON "workflows"("updated_at");
CREATE INDEX "gates_created_at_idx" ON "gates"("created_at");
CREATE INDEX "evidence_created_at_idx" ON "evidence"("created_at");
CREATE INDEX "reports_updated_at_idx" ON "reports"("updated_at");
