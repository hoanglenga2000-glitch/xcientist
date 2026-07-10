export type WorkstationRun = {
  id?: string;
  task_id: string;
  output_dir: string | null;
  status?: string;
  workstation_run?: boolean;
  direct_training_allowed?: boolean;
  official_submission_allowed?: boolean;
  workstation_run_manifest?: string | null;
  artifact_manifest?: string | null;
  best_model?: string | null;
  best_metrics?: Record<string, number> | null;
  accepted?: boolean;
  validation_gate?: { status?: string };
  started_at?: string | null;
  finished_at?: string | null;
};

export type WorkstationTask = {
  id: string;
  name: string;
  task_type: string;
  target?: string | null;
  metric?: string | null;
  status: string;
  priority?: string | null;
  owner?: string | null;
  config_path?: string | null;
  task_dir?: string | null;
  created_at?: string;
  updated_at?: string;
};

export type WorkstationAction = {
  id?: string;
  action: string;
  task_id?: string | null;
  run_id?: string | null;
  message: string;
  artifact?: string | null;
  metadata?: Record<string, unknown> | null;
  at: string;
};

export type TerminalAgentCommand = {
  label: string;
  command: string;
  description: string;
};

export type TerminalAgentSummary = {
  status: "live_events" | "summary_only" | "pending_run" | "no_runs" | string;
  dashboard_url: string;
  evolution_root: string;
  run_count: number;
  completed_run_count: number;
  latest_run_id: string | null;
  latest_run_dir: string | null;
  latest_run_mtime: string | null;
  latest_pending_run_id?: string | null;
  latest_pending_run_dir?: string | null;
  task_id: string | null;
  metric: string;
  metric_direction: string;
  best_exp_id: string;
  best_cv_score: number | null;
  n_iterations: number;
  n_promotions: number;
  events_path: string | null;
  events_present: boolean;
  event_count: number;
  recent_events: Array<Record<string, unknown>>;
  summary_path: string | null;
  summary_present: boolean;
  iterations: Array<Record<string, unknown>>;
  memory_path: string;
  memory_count: number;
  recent_memory: Array<Record<string, unknown>>;
  commands: TerminalAgentCommand[];
  claim_boundary: string;
};

export type ScientistAutopilotSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  selected_task?: string | null;
  mode?: string;
  summary_lines?: string[];
  tool_trace?: Array<{
    tool?: string;
    ok?: boolean;
    status?: string;
    message?: string;
    rationale?: string;
    confidence?: number;
    evidence_signal?: string;
  } & Record<string, unknown>>;
  next_actions?: string[];
  blockers?: string[];
  decision?: Record<string, unknown>;
  artifact_path?: string;
  action_queue?: ScientistActionQueueItem[];
  action_queue_artifact_path?: string;
  human_gate?: Record<string, unknown>;
};

export type ScientistActionQueueItem = {
  id?: string;
  title?: string;
  status?: string;
  command?: string;
  gate?: string;
  why?: string;
  risk?: string;
  rollback_condition?: string;
  expected_artifacts?: string[];
  evidence?: string[];
  autonomy?: string;
};

export type ScientistActionQueueSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  trace_run_id?: string;
  selected_task?: string | null;
  actions?: ScientistActionQueueItem[];
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistNextActionSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  status?: "executed_read_only_tool" | "blocked_by_gate" | "no_ready_action" | string;
  selected_action?: ScientistActionQueueItem | null;
  executed_tool?: string | null;
  tool_result?: Record<string, unknown>;
  message?: string;
  blocked_reason?: string;
  artifact_path?: string;
  action_queue_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistContinuationProgress = {
  updated_at?: string;
  safe_tool?: string;
  tool_ok?: boolean;
  status?: string;
  tool_artifact_path?: string;
  before_remaining_safe_tools?: string[];
  after_remaining_safe_tools?: string[];
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistContinuationStatusSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  status?: "no_continuation" | "needs_more_tools" | "closed" | "invalid_continuation_artifact" | string;
  completion_ratio?: number;
  total_required_tools?: number;
  completed_required_tools?: number;
  remaining_count?: number;
  remaining_safe_tools?: string[];
  executed_or_completed_tools?: string[];
  progress_history?: ScientistContinuationProgress[];
  safe_next_command?: string;
  next_safe_action_command?: string;
  explicit_user_budget_cap?: boolean;
  continuation_artifact_path?: string;
  artifact_path?: string;
  message?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistContinuationResumeStep = {
  index?: number;
  status?: string;
  executed_tool?: string;
  selected_action_id?: string;
  selected_command?: string;
  before_remaining_safe_tools?: string[];
  after_remaining_safe_tools?: string[];
  tool_artifact_path?: string;
  next_action_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistContinuationResumeSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  status?: "not_run" | "no_continuation" | "needs_more_tools" | "closed" | "blocked_by_gate" | "stalled" | string;
  stop_reason?: string;
  max_steps?: number;
  steps_executed?: number;
  steps?: ScientistContinuationResumeStep[];
  initial_status?: Record<string, unknown>;
  final_status?: Record<string, unknown>;
  remaining_safe_tools?: string[];
  executed_tools?: string[];
  message?: string;
  artifact_path?: string;
  continuation_status_artifact_path?: string;
  continuation_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistRecoverySummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  last_goal?: string | null;
  workspace_root?: string;
  guard_path?: string;
  guard_error?: string;
  recovery_block_preview?: string;
  recent_turn_count?: number;
  recent_step_count?: number;
  recent_turns?: Array<Record<string, unknown>>;
  recent_steps?: Array<Record<string, unknown>>;
  latest_loop?: Record<string, unknown> | null;
  latest_next_action?: ScientistNextActionSummary | Record<string, unknown> | null;
  latest_workplan_artifact?: string;
  latest_repair_artifact?: string;
  latest_contract_artifact?: string;
  action_queue_artifact?: string;
  selected_resume_action?: ScientistActionQueueItem | null;
  blockers?: string[];
  resume_commands?: string[];
  recovery_decision?: "not_run" | "blocked_clear_gates" | "resume_from_selected_action" | "refresh_scientist_loop" | string;
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
};

export type ScientistLoopSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  trace_run_id?: string | null;
  generated_at?: string;
  selected_task?: string | null;
  mode?: string;
  stop_reason?: string;
  steps?: Array<Record<string, unknown>>;
  final_autopilot?: Record<string, unknown> | null;
  final_next_action?: ScientistNextActionSummary | Record<string, unknown> | null;
  lesson?: Record<string, unknown> | null;
  artifact_path?: string;
  lessons_path?: string;
  memory_consolidation?: ScientistMemoryConsolidationSummary | Record<string, unknown> | null;
  memory_consolidation_artifact_path?: string;
  memory_path?: string;
  memory_records_added?: number;
  memory_records_total?: number;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
};

export type ScientistLoopLessonsSummary = {
  present?: boolean;
  artifact_path?: string;
  count?: number;
  latest?: Record<string, unknown> | null;
  recent?: Array<Record<string, unknown>>;
};

export type ScientistMemoryConsolidationSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  artifact_path?: string;
  memory_path?: string;
  records_before?: number;
  candidate_records?: number;
  records_added?: number;
  records_total?: number;
  added_memory_ids?: string[];
  source_counts?: Record<string, unknown>;
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

export type ScientistTurnsSummary = {
  present?: boolean;
  artifact_path?: string;
  count?: number;
  latest?: Record<string, unknown> | null;
  recent?: Array<Record<string, unknown>>;
};

export type ScientistStepTraceSummary = {
  present?: boolean;
  artifact_path?: string;
  count?: number;
  latest?: Record<string, unknown> | null;
  recent?: Array<Record<string, unknown>>;
};

export type ScientistStreamEvent = {
  event_id?: string;
  ts?: string;
  source?: string;
  phase?: string;
  step_id?: string;
  tool?: string;
  status?: string;
  message?: string;
  artifact_path?: string;
  gate?: string;
  evidence?: string[];
  no_training_started?: boolean;
};

export type ScientistStreamSummary = {
  present?: boolean;
  generated_at?: string;
  running?: boolean;
  status?: string;
  transport?: "sse" | "polling" | string;
  heartbeat?: string;
  artifact_path?: string;
  event_count?: number;
  latest_event?: ScientistStreamEvent | null;
  recent_events?: ScientistStreamEvent[];
  latest_turn?: Record<string, unknown> | null;
  latest_terminal_turn?: ScientistTerminalTurnSummary | Record<string, unknown> | null;
  turns_count?: number;
  autopilot_status?: Record<string, unknown>;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistAutopilotStatusSummary = {
  present?: boolean;
  artifact_path?: string;
  running?: boolean;
  status?: "not_started" | "running" | "completed" | "failed" | string;
  run_id?: string | null;
  pid?: number | null;
  started_at?: string | null;
  finished_at?: string | null;
  exit_code?: number | null;
  signal?: string | null;
  message?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistCapabilityScore = {
  name?: string;
  score?: number;
  status?: string;
  passed_checks?: string[];
  missing_checks?: string[];
};

export type ScientistCapabilityGap = {
  capability?: string;
  severity?: "critical" | "high" | "medium" | "low" | string;
  score?: number;
  missing_checks?: string[];
};

export type ScientistUpgradeBacklogItem = {
  id?: string;
  title?: string;
  priority?: "P0" | "P1" | "P2" | string;
  status?: string;
  why?: string;
  safe_next_command?: string;
  expected_artifacts?: string[];
  gate?: string;
  no_training_started?: boolean;
};

export type ScientistCapabilityTrendSummary = {
  path?: string;
  records_before?: number;
  records_after?: number;
  previous_score?: number | null;
  current_score?: number;
  score_delta?: number | null;
  latest_readiness?: string;
  recent?: Record<string, unknown>[];
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistSelfAuditSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  last_goal?: string | null;
  overall_score?: number;
  capability_readiness?: string;
  launch_readiness?: string;
  claim_readiness?: Record<string, unknown>;
  capabilities?: ScientistCapabilityScore[];
  gaps?: ScientistCapabilityGap[];
  upgrade_backlog?: ScientistUpgradeBacklogItem[];
  capability_trend?: ScientistCapabilityTrendSummary;
  evidence_sources?: Record<string, unknown>;
  system_blockers?: string[];
  next_safe_commands?: string[];
  artifact_path?: string;
  backlog_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
};

export type ScientistReadinessGateSummary = {
  name?: string;
  status?: "ready" | "blocked" | string;
  ok?: boolean;
  evidence?: string;
  next_action?: string;
};

export type ScientistReadinessArtifactEvidence = {
  name?: string;
  path?: string;
  present?: boolean;
};

export type ScientistReadinessReportSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  schema?: string;
  generated_at?: string;
  selected_task?: string | null;
  overall_score?: number;
  capability_readiness?: string;
  launch_readiness?: string;
  claim_readiness?: Record<string, unknown>;
  execution_readiness?: Record<string, unknown>;
  readiness_matrix?: ScientistReadinessGateSummary[];
  blocking_reasons?: string[];
  recommended_next_commands?: string[];
  artifact_evidence?: ScientistReadinessArtifactEvidence[];
  source_artifacts?: Record<string, unknown>;
  source_summaries?: Record<string, unknown>;
  artifact_path?: string;
  markdown_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
};

export type ScientistCausalDiagnosisNode = {
  id?: string;
  kind?: string;
  status?: string;
  summary?: string;
  evidence?: string[];
};

export type ScientistCausalDiagnosisEdge = {
  from?: string;
  to?: string;
  relation?: string;
};

export type ScientistCausalDiagnosisItem = {
  id?: string;
  summary?: string;
  severity?: string;
  confidence?: number;
  gate?: string;
  evidence?: string[];
};

export type ScientistCausalIntervention = {
  id?: string;
  title?: string;
  safe_next_command?: string;
  gate?: string;
  expected_artifacts?: string[];
  addresses?: string[];
  no_training_started?: boolean;
};

export type ScientistCausalDiagnosisSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  schema?: string;
  generated_at?: string;
  selected_task?: string | null;
  posture?: string;
  symptoms?: ScientistCausalDiagnosisItem[];
  root_causes?: ScientistCausalDiagnosisItem[];
  interventions?: ScientistCausalIntervention[];
  causal_graph?: {
    nodes?: ScientistCausalDiagnosisNode[];
    edges?: ScientistCausalDiagnosisEdge[];
  };
  next_safe_command?: string;
  claim_boundary?: Record<string, unknown>;
  source_summaries?: Record<string, unknown>;
  evidence_refs?: Array<Record<string, unknown>>;
  artifact_path?: string;
  markdown_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
};

export type ScientistStrategyOptimizerRanking = {
  rank?: number;
  id?: string;
  title?: string;
  safe_next_command?: string;
  source?: string;
  gate?: string;
  gate_status?: string;
  status?: string;
  expected_impact?: number;
  evidence_strength?: number;
  cost?: string;
  cost_score?: number;
  risk_level?: string;
  risk_penalty?: number;
  total_score?: number;
  expected_artifacts?: string[];
  evidence?: string[];
  addresses?: string[];
  rationale?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistStrategyOptimizerSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  schema?: string;
  generated_at?: string;
  selected_task?: string | null;
  strategy_posture?: string;
  source_posture?: string;
  selected_strategy?: ScientistStrategyOptimizerRanking | null;
  intervention_ranking?: ScientistStrategyOptimizerRanking[];
  decision_matrix?: Record<string, unknown>;
  next_safe_command?: string;
  next_decision?: Record<string, unknown>;
  source_artifacts?: Record<string, unknown>;
  claim_boundary?: Record<string, unknown>;
  artifact_path?: string;
  markdown_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
  message?: string;
};

export type ScientistContextPacketSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  schema?: string;
  generated_at?: string;
  selected_task?: string | null;
  task_profile?: Record<string, unknown>;
  readiness?: {
    llm_ready?: boolean;
    kaggle_ready?: boolean;
    compute_backend?: string;
    gpu_ready?: boolean;
    gpu_blocked?: boolean;
    can_execute?: boolean;
    blocking_gates?: string[];
    advisory_gaps?: string[];
  };
  active_strategy?: {
    present?: boolean;
    strategy_posture?: string;
    selected_action?: string;
    selected_command?: string;
    gate_status?: string;
    why?: string;
    artifact_path?: string;
  };
  requirement_context?: Record<string, unknown>;
  memory_digest?: {
    retrospective_records?: number;
    retrospective_memory_path?: string;
    scientist_memory_records_added?: number | null;
    scientist_memory_records_total?: number | null;
    scientist_memory_artifact_path?: string;
    recent_lessons?: string[];
  };
  artifact_inventory?: Array<{ name?: string; path?: string; present?: boolean }>;
  context_quality?: {
    score?: number;
    present_artifacts?: number;
    missing_sources?: string[];
    interpretation?: string;
  };
  next_safe_command?: string;
  response_contract?: Record<string, unknown>;
  artifact_path?: string;
  markdown_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
  message?: string;
};

export type ScientistUpgradePlanStep = {
  id?: string;
  step_id?: string;
  backlog_id?: string;
  title?: string;
  priority?: "P0" | "P1" | "P2" | string;
  status?: string;
  files_to_inspect?: string[];
  files_to_edit?: string[];
  expected_artifacts?: string[];
  acceptance_checks?: string[];
  gate?: string;
  safe_next_command?: string;
  no_training_started?: boolean;
};

export type ScientistUpgradePlanSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  readiness?: string;
  source_backlog_path?: string;
  source_self_audit_path?: string;
  open_backlog_count?: number;
  self_audit_score?: number;
  planned_steps?: ScientistUpgradePlanStep[];
  execution_policy?: Record<string, unknown>;
  next_safe_commands?: string[];
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

export type ScientistSelfUpgradeWorkOrder = {
  work_order_id?: string;
  selected_backlog_id?: string;
  title?: string;
  priority?: string;
  objective?: string;
  files_to_edit?: string[];
  files_to_inspect?: string[];
  expected_artifacts?: string[];
  acceptance_checks?: string[];
  rollback_condition?: string;
  claim_boundary?: string;
  code_agent_prompt?: string;
  human_gate?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistSelfUpgradeLoopSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  status?: string;
  selected_backlog_id?: string;
  selected_step_id?: string | null;
  selected_title?: string;
  overall_score_before?: number | null;
  open_backlog_count?: number;
  work_order?: ScientistSelfUpgradeWorkOrder | null;
  action_queue?: Record<string, unknown> | null;
  loop_phases?: Array<Record<string, unknown>>;
  next_safe_commands?: string[];
  artifact_path?: string;
  work_order_path?: string;
  trials_path?: string;
  source_upgrade_plan_path?: string;
  source_self_audit_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
  message?: string;
};

export type ScientistPatchWorkOrderBody = {
  work_order_id?: string;
  issue_id?: string;
  status?: string;
  title?: string;
  objective?: string;
  rationale?: string;
  files_to_edit?: string[];
  files_to_inspect?: string[];
  acceptance_checks?: string[];
  expected_artifacts?: string[];
  rollback_condition?: string;
  claim_boundary?: string;
  safe_next_command?: string;
  code_agent_prompt?: string;
  human_gate?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistPatchActionQueueItem = {
  id?: string;
  title?: string;
  status?: string;
  command?: string;
  work_order_path?: string;
  gate?: string;
  risk?: string;
  no_training_started?: boolean;
};

export type ScientistPatchActionQueueSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  source?: string;
  generated_at?: string;
  actions?: ScientistPatchActionQueueItem[];
  no_training_started?: boolean;
  official_submit?: string;
  artifact_path?: string;
};

export type ScientistPatchWorkOrderSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  status?: string;
  selected_issue_id?: string;
  selected_title?: string;
  evidence?: Record<string, unknown>;
  work_order?: ScientistPatchWorkOrderBody | null;
  action_queue?: ScientistPatchActionQueueSummary | Record<string, unknown> | null;
  next_safe_commands?: string[];
  artifact_path?: string;
  action_queue_path?: string;
  trials_path?: string;
  source_artifacts?: string[];
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
  message?: string;
};

export type ScientistInnovationHypothesis = {
  id?: string;
  strategy_name?: string;
  components?: string[];
  rationale?: string;
  evidence_records?: Array<Record<string, unknown>>;
  risk_controls?: string[];
  expected_artifacts?: string[];
  gate?: string;
  proposed_branch_type?: string;
  code_generation_mode?: string;
  memory_reuse_plan?: Record<string, unknown>;
  ready_for_training?: boolean;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistInnovationBacklogSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  task_profile?: Record<string, unknown>;
  memory_summary?: Record<string, unknown>;
  memory_reuse_plan?: Record<string, unknown>;
  source_paths?: Record<string, unknown>;
  innovation_hypotheses?: ScientistInnovationHypothesis[];
  next_safe_commands?: string[];
  artifact_path?: string;
  innovation_log_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
  message?: string;
};

export type ScientistHypothesisReviewItem = {
  rank?: number;
  hypothesis_id?: string;
  strategy_name?: string;
  branch_type?: string;
  code_generation_mode?: string;
  score?: number;
  evidence_score?: number;
  readiness_score?: number;
  impact_score?: number;
  risk_penalty?: number;
  risk_level?: string;
  status?: string;
  reasons?: string[];
  blockers?: string[];
  next_gate?: string;
  memory_reuse_plan?: Record<string, unknown>;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistHypothesisReviewSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  task_profile?: Record<string, unknown>;
  memory_summary?: Record<string, unknown>;
  memory_reuse_plan?: Record<string, unknown>;
  hypotheses_reviewed?: number;
  reviews?: ScientistHypothesisReviewItem[];
  selected_hypothesis?: ScientistHypothesisReviewItem | null;
  recommendation?: string;
  gate_summary?: Record<string, unknown>;
  next_safe_commands?: string[];
  artifact_path?: string;
  source_backlog_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
  message?: string;
};

export type ScientistExperimentBlueprintBody = {
  blueprint_id?: string;
  task_id?: string;
  hypothesis_id?: string;
  strategy_name?: string;
  branch_type?: string;
  code_generation_mode?: string;
  resource_mode?: string;
  run_command?: string;
  dry_run_command?: string;
  expected_delta?: unknown;
  rollback_condition?: string;
  validation_plan?: string[];
  required_artifacts?: string[];
  promotion_gates?: string[];
  memory_writeback_plan?: Record<string, unknown>;
  memory_reuse_plan?: Record<string, unknown>;
  claim_boundary?: string;
};

export type ScientistExperimentBlueprintSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  blueprint_status?: string;
  selected_hypothesis?: ScientistHypothesisReviewItem | null;
  memory_reuse_plan?: Record<string, unknown>;
  experiment_blueprint?: ScientistExperimentBlueprintBody | null;
  gate_summary?: Record<string, unknown>;
  next_safe_commands?: string[];
  artifact_path?: string;
  source_review_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

export type ScientistSituationModelBlocker = {
  blocker?: string;
  category?: string;
  severity?: string;
  repair_command?: string;
};

export type ScientistSituationModelBody = {
  research_question?: string;
  reasoning_mode?: string;
  posture?: string;
  readiness_score?: number;
  readiness_checks?: Record<string, boolean>;
  evidence_map?: Array<Record<string, unknown>>;
  uncertainties?: string[];
  blocker_model?: ScientistSituationModelBlocker[];
  strategy_model?: Record<string, unknown>;
  self_evolution_model?: Record<string, unknown>;
  recommended_tool_sequence?: string[];
  stop_conditions?: string[];
};

export type ScientistSituationModelSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  situation_status?: string;
  situation_model?: ScientistSituationModelBody | null;
  readiness_score?: number;
  blockers?: string[];
  next_safe_commands?: string[];
  artifact_path?: string;
  source_artifacts?: string[];
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

export type ScientistTurnPlanTool = {
  tool?: string;
  why?: string;
  confidence?: number;
  gate?: string;
  expected_artifacts?: string[];
};

export type ScientistEvidenceGap = {
  severity?: string;
  gap?: string;
  why_it_matters?: string;
  suggested_tool?: string;
};

export type ScientistScientificCritique = {
  decision?: string;
  actionability_score?: number;
  evidence_gaps?: ScientistEvidenceGap[];
  uncertainty_drivers?: string[];
  recommended_tool_inserts?: string[];
  tool_rationale?: Array<Record<string, unknown>>;
  claim_boundaries?: string[];
};

export type ScientistToolBudget = {
  default_max_tools?: number;
  recommended_min_tools?: number;
  max_allowed_tools?: number;
  must_run_tools?: string[];
  expansion_reason?: string;
  completion_gate?: string;
  requested_max_tools?: number;
  effective_max_tools?: number;
  executed_tool_count?: number;
  context_packet_auto_executed?: boolean;
  reasoning_synthesis_auto_executed?: boolean;
  must_run_deferred_count?: number;
};

export type ScientistParityPhase = {
  phase?: "observe" | "plan" | "act" | "reflect" | "improve" | string;
  status?: string;
  purpose?: string;
  evidence?: Record<string, unknown>;
  tool_sequence?: string[];
  gate?: string;
  executed_tools?: string[];
  deferred_tools?: string[];
  evidence_gaps?: string[];
  decision?: string;
  next_safe_command?: string;
  improvement_record?: Record<string, unknown>;
} & Record<string, unknown>;

export type ScientistParityLifecycle = {
  schema?: string;
  loop_name?: string;
  goal?: string;
  intent?: { kind?: string; payload?: string; args?: string[] };
  phases?: ScientistParityPhase[];
  phase_status?: Record<string, string>;
  executed_tools?: string[];
  deferred_tools?: string[];
  must_run_deferred_tools?: string[];
  budget_exhausted?: boolean;
  completion_gate?: Record<string, unknown>;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistRequirementLedger = {
  schema?: string;
  goal?: string;
  intent?: { kind?: string; payload?: string; args?: string[] };
  requirements?: Array<Record<string, unknown>>;
  satisfied_requirements?: string[];
  open_requirements?: string[];
  blocked_requirements?: string[];
  next_evidence_to_collect?: string[];
  completion_gate?: Record<string, unknown>;
};

export type ScientistTurnPlanSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  user_goal?: string;
  intent?: { kind?: string; payload?: string; args?: string[] };
  goal_interpretation?: Record<string, unknown>;
  autonomy_level?: string;
  readiness?: Record<string, unknown>;
  selected_tools?: ScientistTurnPlanTool[];
  tool_sequence?: string[];
  scientific_critique?: ScientistScientificCritique;
  requirement_ledger?: ScientistRequirementLedger;
  tool_budget?: ScientistToolBudget;
  parity_lifecycle?: ScientistParityLifecycle;
  expected_artifacts?: string[];
  stop_conditions?: string[];
  next_safe_command?: string;
  artifact_state?: Record<string, unknown>;
  response_contract?: Record<string, unknown>;
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

export type ScientistTerminalTurnTool = {
  tool?: string;
  ok?: boolean;
  artifact_path?: string;
  message?: string;
};

export type ScientistReasoningHypothesisSummary = {
  id?: string;
  title?: string;
  mechanism?: string;
  falsifiable_prediction?: string;
  required_evidence?: string[];
  experiment?: string;
  success_threshold?: string;
  disconfirming_result?: string;
  evidence_strength?: string;
  risk?: string;
  cost?: string;
  expected_value?: string;
  epistemic_status?: string;
};

export type ScientistReasoningSynthesisSummary = {
  present?: boolean;
  ok?: boolean;
  schema?: string;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  user_goal?: string;
  reasoning_mode?: string;
  direct_answer?: string;
  problem_frame?: Record<string, unknown>;
  hypotheses?: ScientistReasoningHypothesisSummary[];
  comparison?: Array<Record<string, unknown>>;
  selected_hypothesis_id?: string;
  selected_rationale?: string;
  next_safe_action?: {
    action?: string;
    command?: string;
    gate?: string;
    expected_evidence?: string[];
  };
  unresolved_questions?: string[];
  claim_boundaries?: string[];
  answer_markdown?: string;
  reasoning_quality?: {
    score?: number;
    status?: string;
    checks?: Record<string, boolean>;
    missing_contract_items?: string[];
    hypotheses_requested?: number;
    hypotheses_produced?: number;
    complete_falsifiable_hypotheses?: number;
  };
  llm?: {
    used?: boolean;
    provider?: string;
    model?: string;
    input_tokens?: number;
    output_tokens?: number;
    cache_read_tokens?: number;
    error?: string;
  };
  cache_hit?: boolean;
  cache_stats?: {
    requests?: number;
    hits?: number;
    misses?: number;
    hit_ratio?: number;
    last_result?: string;
  };
  cache_stats_path?: string;
  epistemic_status?: string;
  artifact_path?: string;
  markdown_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistTerminalTurnSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  user_goal?: string;
  plan_artifact_path?: string;
  autonomy_level?: string;
  tool_sequence?: string[];
  executed_tools?: ScientistTerminalTurnTool[];
  scientific_critique?: ScientistScientificCritique;
  requirement_ledger?: ScientistRequirementLedger;
  tool_budget?: ScientistToolBudget;
  deferred_tools?: string[];
  must_run_deferred_tools?: string[];
  budget_exhausted?: boolean;
  parity_lifecycle?: ScientistParityLifecycle;
  parity_loop_artifact?: string;
  next_safe_command?: string;
  stop_conditions?: string[];
  execution_ready?: boolean;
  execution_blocked?: boolean;
  blocking_gates?: string[];
  reasoning_synthesis?: ScientistReasoningSynthesisSummary;
  answer_markdown?: string;
  reasoning_quality?: ScientistReasoningSynthesisSummary["reasoning_quality"];
  artifacts?: string[];
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

export type ScientistTerminalTurnResponse = {
  ok: boolean;
  action: string;
  cli_result?: Record<string, unknown>;
  scientist_terminal_turn: ScientistTerminalTurnSummary;
  scientist_turn?: ScientistTerminalTurnSummary;
  scientist_reasoning_synthesis?: ScientistReasoningSynthesisSummary | null;
  scientist_context_packet?: ScientistContextPacketSummary | null;
  scientist_strategy_optimizer?: ScientistStrategyOptimizerSummary | null;
  scientist_action_queue?: ScientistActionQueueSummary | null;
  scientist_step_trace?: ScientistStepTraceSummary | null;
  no_training_started?: boolean;
  official_submit?: string;
  error?: string;
};

export type ScientistWorkplanSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  selected_task?: string | null;
  mode?: string;
  autonomy_level?: string;
  current_focus?: Record<string, unknown> | null;
  summary?: Record<string, unknown>;
  steps?: Array<Record<string, unknown>>;
  resume_commands?: string[];
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  claim_boundary?: string;
};

export type ScientistRepairPlanSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  selected_task?: string | null;
  mode?: string;
  diagnosis?: Array<Record<string, unknown>>;
  root_causes?: string[];
  repair_steps?: Array<Record<string, unknown>>;
  safe_next_command?: string;
  decision?: Record<string, unknown>;
  workplan_focus?: Record<string, unknown>;
  latest_run_signal?: Record<string, unknown>;
  step_trace_considered?: Array<Record<string, unknown>>;
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  claim_boundary?: string;
};

export type ScientistExecutionGateDecisionSummary = {
  ok?: boolean;
  blocked?: boolean;
  status?: "not_run" | "ready_for_gated_training" | "blocked" | string;
  require_model_ready?: boolean;
  blocked_by?: string[];
  root_causes?: string[];
  setup_blockers?: string[];
  safe_next_commands?: string[];
  message?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

export type ScientistExecutionContractSummary = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  selected_task?: string | null;
  compute_mode?: string;
  go_no_go?: "not_run" | "go" | "no_go" | "conditional_go_data_contract_first" | string;
  agent_session_ready?: boolean;
  model_training_ready?: boolean;
  root_causes?: string[];
  setup_blockers?: string[];
  data_contract_status?: string;
  decision?: Record<string, unknown>;
  execution_gate_decision?: ScientistExecutionGateDecisionSummary;
  execution_command?: string;
  enriched_goal?: string;
  required_artifacts?: string[];
  rollback_condition?: string;
  risk_controls?: string[];
  linked_artifacts?: Record<string, string>;
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  claim_boundary?: string;
};

export type WorkstationSummary = {
  tasks?: WorkstationTask[];
  runs?: WorkstationRun[];
  actions?: WorkstationAction[];
  gates?: Array<Record<string, unknown>>;
  evidence?: Array<Record<string, unknown>>;
  reports?: Array<Record<string, unknown>>;
  workflows?: Array<Record<string, unknown>>;
  connector_status?: Record<string, unknown>;
  final_delivery_status?: {
    status?: string;
    generated_at?: string;
    ready_mode?: string;
    task_results?: Record<string, { run_id?: string; metric?: string; value?: number; gate?: string }>;
    connectors?: Record<string, { configured?: boolean; state?: string; optional?: boolean; required_env?: string[] }>;
    audited_actions?: string[];
    acceptance_command?: string;
    resource_smoke_command?: string;
  } | null;
  kaggle_new_competition_readiness?: {
    status?: string;
    task_id?: string;
    competition_slug?: string;
    config_path?: string;
    task_type?: string;
    target?: string;
    metric?: string;
    train_rows?: number;
    test_rows?: number;
    sample_submission_rows?: number;
    local_baseline_ready?: boolean;
    official_download?: { status?: string; reason?: string; attempted?: boolean };
    next_commands?: string[];
    conclusion?: string;
  } | null;
  kaggle_dpapi_readiness?: {
    present?: boolean;
    configured?: boolean;
    credential_status?: string;
    token_type?: string;
    token_loaded_in_env?: boolean;
    credential_file_present?: boolean;
    toolchain_ready?: boolean;
    python_package_version?: string | null;
    cli_path?: string | null;
    report_path?: string;
    safe_install_command?: string;
    real_api_smoke_command?: string;
    human_gate_required_for_submission?: boolean;
  } | null;
  kaggle_experiment_inventory?: {
    schema?: string;
    created_at?: string;
    task_count_with_experiments?: number;
    total_runs_observed?: number;
    total_scored_runs?: number;
    total_promoted_runs?: number;
    total_held_runs?: number;
    total_timeout_or_failed_runs?: number;
    governance_artifact_coverage?: Record<string, number>;
    official_submission_records?: Array<Record<string, unknown>>;
    official_top30_count?: number;
    official_top30_rate?: number;
    claim_boundary?: string;
    task_summary?: Array<Record<string, unknown>>;
  } | null;
  top30_next_evolution_orders?: {
    schema?: string;
    created_at?: string;
    rank_target_percentile?: number;
    official_submit_budget_policy?: string;
    claim_boundary?: string;
    orders?: Array<Record<string, unknown>>;
    required_workstation_artifacts?: string[];
  } | null;
  mlevolve_alignment_matrix?: {
    schema?: string;
    created_at?: string;
    source_repo?: string;
    mlevolve_reference_policy?: Record<string, unknown>;
    workstation_mapping?: Record<string, unknown>;
    claim_boundary?: string;
  } | null;
  mlebench_style_leaderboard?: {
    schema?: string;
    created_at?: string;
    target_reference?: Record<string, unknown>;
    summary?: Record<string, unknown>;
    leaderboard_rows?: Array<Record<string, unknown>>;
    claim_boundary?: string;
  } | null;
  verified_launch_audit?: {
    status?: string;
    launch_state?: string | null;
    blockers?: string[];
    critical_failures?: string[];
    soft_failures?: string[];
    latest_readiness?: Record<string, unknown> | null;
    [key: string]: unknown;
  } | null;
  learning_loop_readiness?: {
    status?: string;
    failures?: string[];
    resource_blockers?: string[];
    memory?: { file_count?: number; record_count?: number; files?: string[] };
    search_orders?: { file_count?: number; record_count?: number; files?: string[] };
    training_progress?: {
      tasks_with_experiments?: number;
      observed_runs?: number;
      scored_runs?: number;
      promoted_runs?: number;
      held_runs?: number;
      official_submission_tasks?: number;
      official_top30_tasks?: number;
      medal_count?: number;
      benchmark_claim_status?: string;
    };
    deepseek_cache?: Record<string, unknown>;
    next_run_queue?: {
      status?: string;
      ready_to_start_now?: boolean;
      blockers?: string[];
      queued_count?: number | null;
      recommended_first_batch?: unknown;
    };
    claim_boundary?: string;
  } | null;
  stages?: Array<Record<string, unknown>>;
  runtime?: {
    task_id?: string;
    latest_experiment_dir?: string | null;
    latest_workstation_run_dir?: string | null;
    latest_score_gated_run_dir?: string | null;
    task_state?: Record<string, unknown> | null;
    agent_trace?: Array<Record<string, unknown>>;
    event_log?: Array<Record<string, unknown>>;
    artifact_manifest?: Record<string, unknown> | null;
    evidence_graph?: Record<string, unknown> | null;
    experiment_graph?: Record<string, unknown> | null;
    reflection?: Record<string, unknown> | null;
    memory?: Record<string, unknown> | null;
    gate_engine?: Record<string, unknown> | null;
    runtime_snapshot?: Record<string, unknown> | null;
    report_markdown?: string;
    generated_code?: string;
    training_log?: string[];
  };
  runtime_by_task?: Record<string, NonNullable<WorkstationSummary["runtime"]>>;
  terminal_agent?: TerminalAgentSummary;
  scientist_autopilot?: ScientistAutopilotSummary;
  scientist_action_queue?: ScientistActionQueueSummary;
  scientist_next_action?: ScientistNextActionSummary;
  scientist_recovery?: ScientistRecoverySummary;
  scientist_loop?: ScientistLoopSummary;
  scientist_loop_lessons?: ScientistLoopLessonsSummary;
  scientist_memory_consolidation?: ScientistMemoryConsolidationSummary;
  scientist_self_audit?: ScientistSelfAuditSummary;
  scientist_readiness_report?: ScientistReadinessReportSummary;
  scientist_causal_diagnosis?: ScientistCausalDiagnosisSummary;
  scientist_strategy_optimizer?: ScientistStrategyOptimizerSummary;
  scientist_context_packet?: ScientistContextPacketSummary;
  scientist_upgrade_plan?: ScientistUpgradePlanSummary;
  scientist_self_upgrade_loop?: ScientistSelfUpgradeLoopSummary;
  scientist_patch_work_order?: ScientistPatchWorkOrderSummary;
  scientist_innovation_backlog?: ScientistInnovationBacklogSummary;
  scientist_hypothesis_review?: ScientistHypothesisReviewSummary;
  scientist_experiment_blueprint?: ScientistExperimentBlueprintSummary;
  scientist_situation_model?: ScientistSituationModelSummary;
  scientist_turn_plan?: ScientistTurnPlanSummary;
  scientist_terminal_turn?: ScientistTerminalTurnSummary;
  scientist_reasoning_synthesis?: ScientistReasoningSynthesisSummary;
  scientist_workplan?: ScientistWorkplanSummary;
  scientist_repair_plan?: ScientistRepairPlanSummary;
  scientist_execution_contract?: ScientistExecutionContractSummary;
  scientist_turns?: ScientistTurnsSummary;
  scientist_step_trace?: ScientistStepTraceSummary;
  scientist_autopilot_status?: ScientistAutopilotStatusSummary;
  scientist_continuation_status?: ScientistContinuationStatusSummary;
  workspace_root?: string;
};

export type WorkstationActionRequest = {
  action: string;
  task_id?: string;
  metadata?: Record<string, unknown>;
};

export type UiComponentClickMetadata = {
  page: string;
  component_type: "button" | "link" | "input" | "textarea" | "select" | "table_row" | string;
  action_id?: string;
  label?: string;
  href?: string;
  disabled?: boolean;
};

export type WorkstationActionResponse = WorkstationAction & {
  ok: boolean;
  action_id?: string;
  task_id?: string;
  config_path?: string;
  runnable?: boolean;
  error?: string;
  decision?: string;
  run_id?: string;
  workflow_id?: string;
  manifest_path?: string;
  artifact_manifest_path?: string;
  markdown_path?: string;
  json_path?: string;
  gate_id?: string;
  latest_run?: string | null;
};

export type LiteratureSearchSource = "local" | "arxiv" | "seed";

export type LiteraturePaper = {
  id: string;
  title: string;
  type: string;
  year: string;
  venue: string;
  score: number;
  task: string;
  exp: string;
  status: string;
  source: LiteratureSearchSource;
  url?: string | null;
  artifact_path?: string | null;
  abstract?: string;
  methods?: string[];
  risks?: string[];
  authors?: string[];
};

export type LiteratureChunk = {
  id: string;
  rank: number;
  chunk: string;
  score: number;
  source: string;
  page: string;
  artifact: string;
  used: string;
  paper_id: string;
  method_tags: string[];
  risk_tags: string[];
};

export type LiteratureStrategy = {
  strategy: string;
  paper_id: string;
  family: string;
  exp: string;
  benefit: string;
  risk: string;
};

export type LiteratureClaimAudit = {
  claim: string;
  paper: string;
  exp: string;
  artifact: string;
  status: string;
};

export type LiteratureSearchResponse = {
  ok: boolean;
  task_id: string;
  query: string;
  generated_at: string;
  source_counts: Record<string, number>;
  metrics: {
    paper_count: number;
    chunk_count: number;
    citation_confidence: number;
    context_tokens: number;
    max_tokens: number;
    local_documents_indexed: number;
    arxiv_results: number;
  };
  papers: LiteraturePaper[];
  retrieval: LiteratureChunk[];
  strategies: LiteratureStrategy[];
  claim_audit: LiteratureClaimAudit[];
  context_markdown: string;
  context_path: string;
  manifest_path: string;
  used_fallback: boolean;
  error?: string;
};

export type RunLocalExperimentResponse = {
  ok: boolean;
  task_id: string;
  run_id?: string;
  experiment_dir?: string;
  validation?: Record<string, unknown>;
  summary?: WorkstationSummary;
  error?: string;
};

export type ClaudeAgentSessionResponse = {
  ok: boolean;
  configured: boolean;
  session_id: string;
  task_id: string;
  status: string;
  provider: string;
  model: string;
  prompt_summary: string;
  transcript_path: string;
  manifest_path: string;
  patch_path: string | null;
  generated_code: string;
  patch_diff: string;
  usage?: Record<string, unknown>;
  prompt_cache_hit_tokens?: number | null;
  deepseek_cache?: Record<string, unknown>;
  prompt_fingerprint?: string;
  cache_key?: string;
  local_response_cache_hit?: boolean;
  cache_entry_path?: string | null;
  external_model_calls_allowed?: boolean;
  missing_env?: string[];
  error?: string;
};

export type GpuGatewayResponse = {
  ok: boolean;
  configured: boolean;
  status: string;
  provider: string;
  missing_env?: string[];
  host?: string;
  username?: string;
  remote_workspace?: string;
  artifact_path?: string;
  stdout_artifact?: string;
  stderr_artifact?: string;
  job_manifest_path?: string;
  artifact_manifest_path?: string;
  pulled_artifacts?: string[];
  allowed_templates?: string[];
  required_fields?: string[];
  stdout?: string;
  stderr?: string;
  error?: string;
};

export type DeepSeekSmokeResponse = {
  ok: boolean;
  configured: boolean;
  provider: "deepseek";
  status: "not_configured" | "passed" | "failed";
  model: string;
  base_url: string;
  content?: string;
  usage?: Record<string, unknown>;
  artifact_path?: string;
  missing_env?: string[];
  error?: string;
};

export type PaperEvidenceBundle = {
  schema: string;
  generated_at: string;
  title: string;
  source_summaries: string[];
  figure_manifest?: string;
  benchmark_results: string;
  claim_boundary: {
    allowed: string[];
    not_allowed: string[];
  };
  latest_round?: string;
  headline_results: {
    tasks: number;
    round2_promoted: number;
    round3_promoted: number;
    round4_promoted?: number;
    round4_preserved_parent?: number;
    best_so_far_never_regressed: boolean;
  };
  steady_improvement_protocol?: {
    paper_core_claim?: string;
    monotonicity_certificate?: {
      all_tasks_best_so_far_never_regressed?: boolean;
      tasks_total?: number;
      tasks_final_improved_over_round1?: number;
      round3_promoted_tasks?: number;
      round3_preserved_parent_tasks?: number;
    };
    next_round_plan?: string;
  };
  steady_improvement_verification?: {
    status?: string;
    claim_verified?: string;
    checks?: Array<{ id: string; description?: string; passed: boolean; evidence?: unknown }>;
    report_path?: string;
  };
  round4_search_plan?: {
    branches?: Array<{
      task_id: string;
      search_stage?: string;
      branch_type?: string;
      code_generation_mode?: string;
      parent_best_score?: number;
      metric?: string;
      direction?: string;
      hypothesis?: string;
      rollback_condition?: string[];
    }>;
  };
  round4_summary?: Record<string, unknown> | null;
  round4_memory?: Record<string, unknown> | null;
  figure_manifest_payload?: {
    figures?: Array<{
      figure_id: string;
      title: string;
      caption?: string;
      paths?: { png?: string; svg?: string };
    }>;
  };
  active_trajectory?: Array<{
    task_id: string;
    metric: string;
    direction: "minimize" | "maximize" | string;
    round1_baseline: number;
    round2_best_so_far: number;
    round3_best_so_far?: number;
    round4_score?: number;
    round4_decision?: string;
    final_best_so_far: number;
    output_dir: string;
    branch_id: string;
    claim_audit: string;
    validation_contract: string;
  }>;
  round4_trajectory?: Array<{
    task_id: string;
    metric: string;
    direction: "minimize" | "maximize" | string;
    round1_baseline: number;
    round2_best_so_far: number;
    round3_best_so_far: number;
    round4_score: number;
    round4_decision: string;
    final_best_so_far: number;
    output_dir: string;
    branch_id: string;
    claim_audit: string;
    validation_contract: string;
  }>;
  trajectory: Array<{
    task_id: string;
    metric: string;
    direction: "minimize" | "maximize" | string;
    round1_baseline: number;
    round2_best_so_far: number;
    round3_score: number;
    round3_decision: string;
    final_best_so_far: number;
    output_dir: string;
    branch_id: string;
    claim_audit: string;
    validation_contract: string;
  }>;
};

export type PaperEvidenceBundleResponse = {
  ok: boolean;
  bundle_path?: string;
  bundle?: PaperEvidenceBundle;
  run_id?: string;
  paper_report?: string;
  benchmark_results?: string;
  summary?: WorkstationSummary;
  error?: string;
};

// ── Evolution engine (self-evolving research brain) ──────────────────────────
export type EvolutionGraphNode = {
  exp_id: string;
  parent_id: string | null;
  branch_type: string;
  cv_score: number | null;
  public_score?: number | null;
  promoted: boolean;
  decision?: string;
  implementation_summary?: string;
  hypothesis?: string;
  risk_flags?: string[];
  metric_name?: string;
  metric_direction?: string;
  promotion_reason?: string;
  created_at?: string;
};

export type EvolutionGraphEdge = { source: string; target: string; reason?: string };

export type EvolutionGraphResponse = {
  ok: boolean;
  task_id: string;
  node_count: number;
  has_run: boolean;
  nodes: EvolutionGraphNode[];
  edges: EvolutionGraphEdge[];
  reference_edges?: EvolutionGraphEdge[];
  top_candidates: string[];
  branch_diverse_top_candidates?: string[];
  stagnation_branches?: string[];
  global_stagnation?: boolean;
  selected_next_branch?: string | null;
  exploration_stage?: string;
  metric_name?: string;
  metric_direction?: string;
  best_exp_id?: string | null;
  claim_boundary?: string;
  error?: string;
};

export type EvolutionStateResponse = {
  ok: boolean;
  task_id: string;
  current_stage: string;
  has_run: boolean;
  best_so_far: {
    exp_id: string | null;
    cv_score: number | null;
    metric: string;
    metric_direction: string;
    promotion_reason?: string;
  };
  search_graph_summary: {
    node_count: number;
    edge_count: number;
    top_candidates: string[];
    exploration_stage: string;
    global_stagnation: boolean;
  };
  active_branches: Array<{ exp_id: string; branch_type: string; cv_score: number | null; promoted: boolean }>;
  latest_decision?: string | null;
  memory_hits: number;
  risk_flags: string[];
  gate_status?: string;
  last_artifacts: string[];
  claim_boundary: string;
  official_submit_allowed: boolean;
  generated_at?: string;
  error?: string;
};

export type EvolutionPlanRequest = {
  task_id: string;
  objective?: string;
  budget?: number;
  resource_mode?: string;
  rank_target_percentile?: number | null;
  official_submit_allowed?: false;
  modality?: string;
  task_type?: string;
  metric?: string;
  metric_direction?: string;
  n_train?: number;
  n_features?: number;
  literature_context_path?: string;
};

export type EvolutionPlanResponse = {
  ok: boolean;
  task_id: string;
  objective?: string;
  budget?: number;
  resource_mode?: string;
  rank_target_percentile?: number | null;
  official_submit_allowed: boolean;
  search_controller_decision: string;
  selected_branch: string;
  code_generation_mode: string;
  expansion_type: string;
  reference_exp_ids: string[];
  phase?: string;
  recommended_strategies: string[];
  strategy_rationale?: Record<string, string>;
  expected_delta: string;
  parent_best_score?: number | null;
  rollback_condition: string[];
  validation_contract_path: string;
  plan_path?: string;
  claim_boundary: string;
  generated_at?: string;
  error?: string;
};

export type EvolutionStepResponse = {
  ok: boolean;
  task_id: string;
  dry_run: boolean;
  exp_id?: string;
  decision: string;
  gate_status: string;
  code_generation_mode?: string;
  expansion_type?: string;
  parent_exp_id?: string | null;
  node_count?: number;
  artifacts?: string[];
  reason?: string;
  next_action?: string;
  claim_boundary: string;
  official_submit_allowed?: boolean;
  generated_at?: string;
  error?: string;
};

export type EvolutionMemoryRecord = {
  memory_id: string;
  task_type: string;
  method: string;
  what_worked: string;
  what_failed: string;
  metric_delta: number | null;
  reusable_strategy: string;
  failure_pattern: string;
  linked_exp_ids: string[];
};

export type EvolutionMemoryResponse = {
  ok: boolean;
  task_id: string;
  task_type: string;
  record_count: number;
  memory: EvolutionMemoryRecord[];
  reusable_strategies: string[];
  failure_patterns: string[];
  memory_store: string;
  claim_boundary: string;
  generated_at?: string;
  error?: string;
};

// ── Evolution engine: real task configs (configs/evolution/*.json) ────────────
export type EvolutionConfigSummary = {
  task_id: string; // config file stem; a valid task_id for the run CLI (exact match)
  task_name: string; // Kaggle competition slug
  modality?: string;
  task_type?: string;
  metric?: string;
  metric_direction?: string;
  n_train?: number | null;
  n_features?: number | null;
  has_gpu_data_dir: boolean; // true when the config carries an HPC/GPU data path
  has_local_data_dir: boolean; // true when the config can feed the local runner
};

export type EvolutionConfigsResponse = {
  ok: boolean;
  count: number;
  configs: EvolutionConfigSummary[];
  config_dir: string;
  error?: string;
};

// ── Evolution engine: full closed-loop cycle (plan -> approve -> train -> ingest)
export type EvolutionEngine = "legacy" | "research_os";
export type EvolutionRunner = "gpu" | "local";

export type EvolutionCycleRequest = {
  task_id: string;
  engine?: EvolutionEngine; // UI selects research_os; server code default stays legacy
  runner?: EvolutionRunner; // gpu is the only unblocked runner today
  iterations?: number;
  mcgs?: boolean;
  approve?: boolean; // real training only launches when true (human gate)
  official_submit_allowed?: false;
};

export type EvolutionCycleResponse = {
  ok: boolean;
  task_id: string;
  stage: "awaiting_approval" | "training" | "training_blocked" | "training_failed" | "completed";
  approved: boolean;
  plan?: EvolutionPlanResponse;
  training?: { run_id?: string; best_score?: number | null; nodes_evaluated?: number | null };
  ingest?: Record<string, unknown>;
  best_so_far?: Record<string, unknown>;
  next_action?: string;
  reason?: string;
  official_submit_allowed: boolean;
  claim_boundary?: string;
  error?: string;
};
