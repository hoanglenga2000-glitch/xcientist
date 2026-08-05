import { NextResponse } from "next/server";
import { ensureWorkstationSeeded } from "@/lib/server/bootstrap";
import { logAction } from "@/lib/server/actions";
import { normalizeTaskId } from "@/lib/server/paths";
import { generateScientificReport, lookupScientificReport, readScientificReportGenerationStatus } from "@/lib/server/scientific-report";

export const dynamic = "force-dynamic";

class InvalidReportRequestError extends Error {}

function validatedId(value: string | null, label: string) {
  if (!value || !/^[A-Za-z0-9_-]{1,160}$/.test(value)) throw new InvalidReportRequestError(`${label} is required and must be a valid ID.`);
  return value;
}

export async function GET(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  try {
    await ensureWorkstationSeeded();
    const { taskId: rawTaskId } = await params;
    const taskId = validatedId(normalizeTaskId(rawTaskId), "task_id");
    const searchParams = new URL(request.url).searchParams;
    const runId = validatedId(searchParams.get("run_id"), "run_id");
    if (searchParams.get("status_only") === "1") {
      const generation = await readScientificReportGenerationStatus(taskId, runId);
      return NextResponse.json({ ok: true, task_id: taskId, run_id: runId, generation });
    }
    const lookup = await lookupScientificReport(taskId, runId);
    if (lookup.status === "pending_report") {
      return NextResponse.json({
        ok: false,
        code: "pending_report",
        task_id: taskId,
        run_id: runId,
        status: "pending_report",
        run_status: lookup.runStatus,
        generation: lookup.generation,
        error: "Scientific report is pending reviewed run completion.",
      }, { status: 202 });
    }
    if (lookup.status === "generation_failed") {
      return NextResponse.json({
        ok: false,
        code: "report_generation_failed",
        task_id: taskId,
        run_id: runId,
        status: "failed",
        run_status: lookup.runStatus,
        generation: lookup.generation,
        error: lookup.error,
      }, { status: 422 });
    }
    if (lookup.status === "integrity_failed") {
      return NextResponse.json({
        ok: false,
        code: "report_integrity_failed",
        task_id: taskId,
        run_id: runId,
        status: "failed",
        error: lookup.error,
      }, { status: 409 });
    }
    if (lookup.status === "not_found") {
      return NextResponse.json({ ok: false, task_id: taskId, error: "Scientific report not found for the requested task and run." }, { status: 404 });
    }
    const report = lookup.report;
    if (report.task_id !== taskId || report.run_id !== runId) {
      return NextResponse.json({ ok: false, code: "report_binding_drift", task_id: taskId, run_id: runId, error: "Scientific report binding drift was detected." }, { status: 409 });
    }
    return NextResponse.json({ ok: true, task_id: taskId, report });
  } catch (error) {
    const invalid = error instanceof InvalidReportRequestError;
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Scientific report lookup failed." },
      { status: invalid ? 400 : 500 },
    );
  }
}

export async function POST(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  try {
    await ensureWorkstationSeeded();
    const { taskId: rawTaskId } = await params;
    const taskId = validatedId(normalizeTaskId(rawTaskId), "task_id");
    const body = await request.json().catch(() => ({})) as Record<string, unknown>;
    const runId = validatedId(typeof body.run_id === "string" ? body.run_id : null, "run_id");
    const existing = await lookupScientificReport(taskId, runId);
    if (existing.status === "ready") {
      return NextResponse.json({ ok: true, task_id: taskId, report: existing.report });
    }
    const report = await generateScientificReport(taskId, runId);
    if (report.task_id !== taskId || report.run_id !== runId) throw new Error("Scientific report binding mismatch.");
    await logAction({
      action: "generate_scientific_report",
      taskId,
      message: "Nature Skills scientific report rendered from reviewed run evidence.",
      artifactPath: report.manifest_path,
      metadata: {
        run_id: report.run_id,
        version: report.version,
        renderer: report.renderer,
        status: report.status,
        figure_count: report.figures.filter((figure) => figure.status === "ready").length,
        pdf_ready: Boolean(report.report_pdf_path),
        bundle_ready: Boolean(report.bundle_path)
      }
    });
    return NextResponse.json({ ok: true, task_id: taskId, report });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "scientific report generation failed" }, { status: 422 });
  }
}
