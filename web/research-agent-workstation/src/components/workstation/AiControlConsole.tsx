"use client";

import { useEffect, useRef, useState } from "react";
import {
  Bot,
  BrainCircuit,
  Cpu,
  FileCheck2,
  GitBranch,
  History,
  Lightbulb,
  Play,
  RefreshCcw,
  Send,
  ShieldCheck,
  TerminalSquare,
  Upload,
  XCircle
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge, type StatusTone } from "@/components/ui/status-badge";
import * as api from "@/lib/api/client";
import type { ScientistUpgradeCampaignSummary } from "@/lib/api/types";
import { JsonInspector } from "./Common";

type Locale = "zh-CN" | "en-US";

type ControlIntent =
  | "scientist_autopilot"
  | "scientist_self_audit"
  | "scientist_readiness_report"
  | "scientist_causal_diagnosis"
  | "scientist_strategy_optimizer"
  | "scientist_context_packet"
  | "scientist_upgrade_plan"
  | "scientist_self_upgrade_loop"
  | "scientist_upgrade_campaign"
  | "scientist_patch_work_order"
  | "scientist_innovation_backlog"
  | "scientist_hypothesis_review"
  | "scientist_experiment_blueprint"
  | "scientist_situation_model"
  | "scientist_turn"
  | "scientist_turn_plan"
  | "scientist_next_action"
  | "scientist_continuation_status"
  | "scientist_continuation_resume"
  | "scientist_recovery"
  | "scientist_loop"
  | "scientist_workplan"
  | "scientist_repair_plan"
  | "scientist_execution_contract"
  | "create_workstation_run"
  | "onboard_playground_s6e6"
  | "prepare_hpc_execution_gate"
  | "export_code_agent_context"
  | "deepseek_code_draft"
  | "claude_code_draft"
  | "deepseek_smoke"
  | "gpu_smoke"
  | "gpu_probe_job"
  | "run_local_experiment"
  | "generate_report_draft"
  | "generate_teacher_evidence_bundle"
  | "kaggle_submit"
  | "unknown";

type RiskLevel = "safe" | "gated" | "blocked";

type ParsedControlCommand = {
  intent: ControlIntent;
  taskId: string;
  metadata: Record<string, unknown>;
  risk: RiskLevel;
  blockedReason?: string;
  description: string;
};

type ControlMessage = {
  role: "user" | "system" | "error";
  content: string;
  timestamp: number;
};

type ScientistAutopilotView = {
  present?: boolean;
  mode?: string;
  trace_run_id?: string;
  selected_task?: string | null;
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
  action_queue?: ScientistActionQueueItemView[];
  action_queue_artifact_path?: string;
  blockers?: string[];
  decision?: Record<string, unknown>;
  artifact_path?: string;
  human_gate?: Record<string, unknown>;
};

type ScientistCapabilityScoreView = {
  name?: string;
  score?: number;
  status?: string;
  passed_checks?: string[];
  missing_checks?: string[];
};

type ScientistCapabilityGapView = {
  capability?: string;
  severity?: string;
  score?: number;
  missing_checks?: string[];
};

type ScientistUpgradeBacklogItemView = {
  id?: string;
  title?: string;
  priority?: string;
  status?: string;
  why?: string;
  safe_next_command?: string;
  expected_artifacts?: string[];
  gate?: string;
};

type ScientistCapabilityTrendView = {
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

type ScientistSelfAuditView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  overall_score?: number;
  capability_readiness?: string;
  launch_readiness?: string;
  claim_readiness?: Record<string, unknown>;
  capabilities?: ScientistCapabilityScoreView[];
  gaps?: ScientistCapabilityGapView[];
  upgrade_backlog?: ScientistUpgradeBacklogItemView[];
  capability_trend?: ScientistCapabilityTrendView;
  evidence_sources?: Record<string, unknown>;
  system_blockers?: string[];
  next_safe_commands?: string[];
  artifact_path?: string;
  backlog_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistInnovationHypothesisView = {
  id?: string;
  strategy_name?: string;
  components?: string[];
  rationale?: string;
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

type ScientistReadinessGateView = {
  name?: string;
  status?: string;
  ok?: boolean;
  evidence?: string;
  next_action?: string;
};

type ScientistReadinessReportView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  overall_score?: number;
  capability_readiness?: string;
  launch_readiness?: string;
  claim_readiness?: Record<string, unknown>;
  execution_readiness?: Record<string, unknown>;
  readiness_matrix?: ScientistReadinessGateView[];
  blocking_reasons?: string[];
  recommended_next_commands?: string[];
  artifact_evidence?: { name?: string; path?: string; present?: boolean }[];
  source_summaries?: Record<string, unknown>;
  artifact_path?: string;
  markdown_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistCausalDiagnosisItemView = {
  id?: string;
  summary?: string;
  severity?: string;
  confidence?: number;
  gate?: string;
  evidence?: string[];
};

type ScientistCausalInterventionView = {
  id?: string;
  title?: string;
  safe_next_command?: string;
  gate?: string;
  expected_artifacts?: string[];
  addresses?: string[];
};

type ScientistCausalDiagnosisView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  posture?: string;
  symptoms?: ScientistCausalDiagnosisItemView[];
  root_causes?: ScientistCausalDiagnosisItemView[];
  interventions?: ScientistCausalInterventionView[];
  causal_graph?: {
    nodes?: Array<Record<string, unknown>>;
    edges?: Array<{ from?: string; to?: string; relation?: string }>;
  };
  next_safe_command?: string;
  claim_boundary?: Record<string, unknown>;
  source_summaries?: Record<string, unknown>;
  artifact_path?: string;
  markdown_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistStrategyRankingView = {
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
  risk_level?: string;
  risk_penalty?: number;
  total_score?: number;
  expected_artifacts?: string[];
  evidence?: string[];
  addresses?: string[];
  rationale?: string;
};

type ScientistStrategyOptimizerView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  strategy_posture?: string;
  source_posture?: string;
  selected_strategy?: ScientistStrategyRankingView | null;
  intervention_ranking?: ScientistStrategyRankingView[];
  decision_matrix?: Record<string, unknown>;
  next_safe_command?: string;
  next_decision?: Record<string, unknown>;
  source_artifacts?: Record<string, unknown>;
  artifact_path?: string;
  markdown_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistContextPacketView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
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
  memory_digest?: {
    retrospective_records?: number;
    scientist_memory_records_added?: number | null;
    scientist_memory_records_total?: number | null;
    recent_lessons?: string[];
  };
  requirement_context?: Record<string, unknown>;
  artifact_inventory?: Array<{ name?: string; path?: string; present?: boolean }>;
  context_quality?: {
    score?: number;
    present_artifacts?: number;
    missing_sources?: string[];
    interpretation?: string;
  };
  next_safe_command?: string;
  artifact_path?: string;
  markdown_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistUpgradePlanStepView = {
  id?: string;
  step_id?: string;
  backlog_id?: string;
  title?: string;
  priority?: string;
  status?: string;
  files_to_inspect?: string[];
  files_to_edit?: string[];
  expected_artifacts?: string[];
  acceptance_checks?: string[];
  gate?: string;
  safe_next_command?: string;
};

type ScientistUpgradePlanView = {
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
  planned_steps?: ScientistUpgradePlanStepView[];
  execution_policy?: Record<string, unknown>;
  next_safe_commands?: string[];
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

type ScientistSelfUpgradeWorkOrderView = {
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

type ScientistSelfUpgradeLoopView = {
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
  work_order?: ScientistSelfUpgradeWorkOrderView | null;
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

type ScientistPatchWorkOrderBodyView = {
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

type ScientistPatchActionQueueItemView = {
  id?: string;
  title?: string;
  status?: string;
  command?: string;
  work_order_path?: string;
  gate?: string;
  risk?: string;
  no_training_started?: boolean;
};

type ScientistPatchActionQueueView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  source?: string;
  generated_at?: string;
  actions?: ScientistPatchActionQueueItemView[];
  no_training_started?: boolean;
  official_submit?: string;
  artifact_path?: string;
};

type ScientistPatchWorkOrderView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  status?: string;
  selected_issue_id?: string;
  selected_title?: string;
  evidence?: Record<string, unknown>;
  work_order?: ScientistPatchWorkOrderBodyView | null;
  action_queue?: ScientistPatchActionQueueView | Record<string, unknown> | null;
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

type ScientistEngineeringLoopView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  run_id?: string;
  selected_task?: string | null;
  status?: string;
  message?: string;
  work_order_path?: string;
  work_order?: {
    id?: string;
    title?: string;
    files_to_edit?: string[];
    rollback_condition?: string;
    human_gate?: string;
  };
  patch_path?: string;
  changed_files?: string[];
  patch_applied_in_isolated_worktree?: boolean;
  acceptance_checks?: Array<{
    command?: string;
    allowed?: boolean;
    exit_code?: number | null;
    passed?: boolean;
    log_path?: string;
    output_tail?: string;
  }>;
  candidate_diff_path?: string;
  cleanup_ok?: boolean;
  main_head_before?: string;
  main_head_after?: string;
  main_worktree_modified?: boolean;
  merge_ready?: boolean;
  next_safe_command?: string;
  epistemic_status?: string;
  human_gate?: string;
  artifact_path?: string;
  run_manifest_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistInnovationBacklogView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  task_profile?: Record<string, unknown>;
  memory_summary?: Record<string, unknown>;
  memory_reuse_plan?: Record<string, unknown>;
  source_paths?: Record<string, unknown>;
  innovation_hypotheses?: ScientistInnovationHypothesisView[];
  next_safe_commands?: string[];
  artifact_path?: string;
  innovation_log_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

type ScientistHypothesisReviewItemView = {
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

type ScientistHypothesisReviewView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  task_profile?: Record<string, unknown>;
  memory_summary?: Record<string, unknown>;
  memory_reuse_plan?: Record<string, unknown>;
  hypotheses_reviewed?: number;
  reviews?: ScientistHypothesisReviewItemView[];
  selected_hypothesis?: ScientistHypothesisReviewItemView | null;
  recommendation?: string;
  gate_summary?: Record<string, unknown>;
  next_safe_commands?: string[];
  artifact_path?: string;
  source_backlog_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

type ScientistExperimentBlueprintBodyView = {
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

type ScientistExperimentBlueprintView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  blueprint_status?: string;
  selected_hypothesis?: ScientistHypothesisReviewItemView | null;
  memory_reuse_plan?: Record<string, unknown>;
  experiment_blueprint?: ScientistExperimentBlueprintBodyView | null;
  gate_summary?: Record<string, unknown>;
  next_safe_commands?: string[];
  artifact_path?: string;
  source_review_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

type ScientistSituationModelBlockerView = {
  blocker?: string;
  category?: string;
  severity?: string;
  repair_command?: string;
};

type ScientistSituationModelBodyView = {
  research_question?: string;
  reasoning_mode?: string;
  posture?: string;
  readiness_score?: number;
  readiness_checks?: Record<string, boolean>;
  evidence_map?: Array<Record<string, unknown>>;
  uncertainties?: string[];
  blocker_model?: ScientistSituationModelBlockerView[];
  strategy_model?: Record<string, unknown>;
  self_evolution_model?: Record<string, unknown>;
  recommended_tool_sequence?: string[];
  stop_conditions?: string[];
};

type ScientistSituationModelView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  situation_status?: string;
  situation_model?: ScientistSituationModelBodyView | null;
  readiness_score?: number;
  blockers?: string[];
  next_safe_commands?: string[];
  artifact_path?: string;
  source_artifacts?: string[];
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

type ScientistTurnPlanToolView = {
  tool?: string;
  why?: string;
  confidence?: number;
  gate?: string;
  expected_artifacts?: string[];
};

type ScientistEvidenceGapView = {
  severity?: string;
  gap?: string;
  why_it_matters?: string;
  suggested_tool?: string;
};

type ScientistScientificCritiqueView = {
  decision?: string;
  actionability_score?: number;
  evidence_gaps?: ScientistEvidenceGapView[];
  uncertainty_drivers?: string[];
  recommended_tool_inserts?: string[];
  tool_rationale?: Array<Record<string, unknown>>;
  claim_boundaries?: string[];
};

type ScientistToolBudgetView = {
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

type ScientistParityPhaseView = {
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

type ScientistParityLifecycleView = {
  schema?: string;
  loop_name?: string;
  goal?: string;
  intent?: { kind?: string; payload?: string; args?: string[] };
  phases?: ScientistParityPhaseView[];
  phase_status?: Record<string, string>;
  executed_tools?: string[];
  deferred_tools?: string[];
  must_run_deferred_tools?: string[];
  budget_exhausted?: boolean;
  completion_gate?: Record<string, unknown>;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistRequirementLedgerView = {
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

type ScientistTurnPlanView = {
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
  selected_tools?: ScientistTurnPlanToolView[];
  tool_sequence?: string[];
  scientific_critique?: ScientistScientificCritiqueView;
  requirement_ledger?: ScientistRequirementLedgerView;
  tool_budget?: ScientistToolBudgetView;
  parity_lifecycle?: ScientistParityLifecycleView;
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

type ScientistTerminalTurnToolView = {
  tool?: string;
  ok?: boolean;
  artifact_path?: string;
  message?: string;
};

type ScientistReasoningHypothesisView = {
  id?: string;
  title?: string;
  mechanism?: string;
  falsifiable_prediction?: string;
  experiment?: string;
  success_threshold?: string;
  disconfirming_result?: string;
  evidence_strength?: string;
  risk?: string;
  cost?: string;
  expected_value?: string;
};

type ScientistReasoningSynthesisView = {
  present?: boolean;
  ok?: boolean;
  reasoning_mode?: string;
  direct_answer?: string;
  hypotheses?: ScientistReasoningHypothesisView[];
  comparison?: Array<Record<string, unknown>>;
  selected_hypothesis_id?: string;
  selected_rationale?: string;
  next_safe_action?: {
    action?: string;
    command?: string;
    gate?: string;
    expected_evidence?: string[];
  };
  claim_boundaries?: string[];
  answer_markdown?: string;
  reasoning_quality?: {
    score?: number;
    status?: string;
    hypotheses_requested?: number;
    hypotheses_produced?: number;
    complete_falsifiable_hypotheses?: number;
    missing_contract_items?: string[];
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

type ScientistTerminalTurnView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  user_goal?: string;
  plan_artifact_path?: string;
  autonomy_level?: string;
  tool_sequence?: string[];
  executed_tools?: ScientistTerminalTurnToolView[];
  scientific_critique?: ScientistScientificCritiqueView;
  requirement_ledger?: ScientistRequirementLedgerView;
  tool_budget?: ScientistToolBudgetView;
  deferred_tools?: string[];
  must_run_deferred_tools?: string[];
  budget_exhausted?: boolean;
  parity_lifecycle?: ScientistParityLifecycleView;
  parity_loop_artifact?: string;
  next_safe_command?: string;
  stop_conditions?: string[];
  execution_ready?: boolean;
  execution_blocked?: boolean;
  blocking_gates?: string[];
  reasoning_synthesis?: ScientistReasoningSynthesisView;
  answer_markdown?: string;
  reasoning_quality?: ScientistReasoningSynthesisView["reasoning_quality"];
  artifacts?: string[];
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  message?: string;
};

type ScientistActionQueueItemView = {
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
  metadata?: Record<string, unknown>;
};

type ScientistActionQueueView = {
  present?: boolean;
  artifact_path?: string;
  trace_run_id?: string;
  selected_task?: string | null;
  actions?: ScientistActionQueueItemView[];
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistNextActionView = {
  present?: boolean;
  artifact_path?: string;
  generated_at?: string;
  selected_task?: string | null;
  status?: string;
  selected_action?: ScientistActionQueueItemView | null;
  executed_tool?: string | null;
  message?: string;
  blocked_reason?: string;
  action_queue_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistContinuationProgressView = {
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

type ScientistContinuationStatusView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  status?: string;
  completion_ratio?: number;
  total_required_tools?: number;
  completed_required_tools?: number;
  remaining_count?: number;
  remaining_safe_tools?: string[];
  executed_or_completed_tools?: string[];
  progress_history?: ScientistContinuationProgressView[];
  safe_next_command?: string;
  next_safe_action_command?: string;
  explicit_user_budget_cap?: boolean;
  continuation_artifact_path?: string;
  artifact_path?: string;
  message?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistContinuationResumeStepView = {
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

type ScientistContinuationResumeView = {
  present?: boolean;
  ok?: boolean;
  tool?: string;
  generated_at?: string;
  selected_task?: string | null;
  status?: string;
  stop_reason?: string;
  max_steps?: number;
  steps_executed?: number;
  steps?: ScientistContinuationResumeStepView[];
  remaining_safe_tools?: string[];
  executed_tools?: string[];
  message?: string;
  artifact_path?: string;
  continuation_status_artifact_path?: string;
  continuation_artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistRecoveryView = {
  present?: boolean;
  artifact_path?: string;
  generated_at?: string;
  selected_task?: string | null;
  last_goal?: string | null;
  recovery_decision?: string;
  recent_turn_count?: number;
  recent_step_count?: number;
  latest_loop?: Record<string, unknown> | null;
  latest_next_action?: Record<string, unknown> | null;
  selected_resume_action?: ScientistActionQueueItemView | null;
  blockers?: string[];
  resume_commands?: string[];
  guard_path?: string;
  latest_workplan_artifact?: string;
  latest_repair_artifact?: string;
  latest_contract_artifact?: string;
  action_queue_artifact?: string;
  recovery_block_preview?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistLoopView = {
  present?: boolean;
  artifact_path?: string;
  lessons_path?: string;
  generated_at?: string;
  selected_task?: string | null;
  trace_run_id?: string | null;
  mode?: string;
  stop_reason?: string;
  steps?: Array<Record<string, unknown>>;
  final_autopilot?: Record<string, unknown> | null;
  final_next_action?: ScientistNextActionView | Record<string, unknown> | null;
  lesson?: Record<string, unknown> | null;
  no_training_started?: boolean;
  official_submit?: string;
  human_gate?: Record<string, unknown>;
  memory_consolidation?: ScientistMemoryConsolidationView | Record<string, unknown> | null;
  memory_consolidation_artifact_path?: string;
  memory_path?: string;
  memory_records_added?: number;
  memory_records_total?: number;
};

type ScientistLoopLessonsView = {
  present?: boolean;
  artifact_path?: string;
  count?: number;
  latest?: Record<string, unknown> | null;
  recent?: Array<Record<string, unknown>>;
};

type ScientistMemoryConsolidationView = {
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

type ScientistTurnsView = {
  present?: boolean;
  artifact_path?: string;
  count?: number;
  latest?: Record<string, unknown> | null;
  recent?: Array<Record<string, unknown>>;
};

type ScientistStepTraceView = {
  present?: boolean;
  artifact_path?: string;
  count?: number;
  latest?: Record<string, unknown> | null;
  recent?: Array<Record<string, unknown>>;
};

type ScientistStreamEventView = {
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

type ScientistStreamView = {
  present?: boolean;
  generated_at?: string;
  running?: boolean;
  status?: string;
  transport?: string;
  heartbeat?: string;
  artifact_path?: string;
  event_count?: number;
  latest_event?: ScientistStreamEventView | null;
  recent_events?: ScientistStreamEventView[];
  latest_turn?: Record<string, unknown> | null;
  latest_terminal_turn?: ScientistTerminalTurnView | Record<string, unknown> | null;
  turns_count?: number;
  autopilot_status?: Record<string, unknown>;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistStreamTransportMode = "connecting" | "sse" | "polling" | "unavailable";

type ScientistStreamPayloadView = {
  scientist_stream?: ScientistStreamView | null;
  scientist_step_trace?: ScientistStepTraceView | null;
  scientist_turns?: ScientistTurnsView | null;
  scientist_terminal_turn?: ScientistTerminalTurnView | null;
  scientist_autopilot_status?: ScientistAutopilotStatusView | null;
};

type ScientistAutopilotStatusView = {
  present?: boolean;
  artifact_path?: string;
  running?: boolean;
  status?: string;
  run_id?: string | null;
  pid?: number | null;
  started_at?: string | null;
  finished_at?: string | null;
  message?: string;
};

type ScientistWorkplanView = {
  present?: boolean;
  mode?: string;
  autonomy_level?: string;
  current_focus?: Record<string, unknown> | null;
  summary?: Record<string, unknown>;
  steps?: Array<Record<string, unknown>>;
  resume_commands?: string[];
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistRepairPlanView = {
  present?: boolean;
  mode?: string;
  diagnosis?: Array<Record<string, unknown>>;
  root_causes?: string[];
  repair_steps?: Array<Record<string, unknown>>;
  safe_next_command?: string;
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  claim_boundary?: string;
};

type ScientistExecutionGateDecisionView = {
  ok?: boolean;
  blocked?: boolean;
  status?: string;
  require_model_ready?: boolean;
  blocked_by?: string[];
  root_causes?: string[];
  setup_blockers?: string[];
  safe_next_commands?: string[];
  message?: string;
  no_training_started?: boolean;
  official_submit?: string;
};

type ScientistExecutionContractView = {
  present?: boolean;
  go_no_go?: string;
  agent_session_ready?: boolean;
  model_training_ready?: boolean;
  data_contract_status?: string;
  root_causes?: string[];
  setup_blockers?: string[];
  decision?: Record<string, unknown>;
  execution_gate_decision?: ScientistExecutionGateDecisionView;
  execution_command?: string;
  required_artifacts?: string[];
  rollback_condition?: string;
  risk_controls?: string[];
  artifact_path?: string;
  no_training_started?: boolean;
  official_submit?: string;
  claim_boundary?: string;
};

type ScreenProps = {
  selectedTask: string;
  locale?: Locale;
  runWorkstationAction?: (action: string, metadata?: Record<string, unknown>) => Promise<unknown>;
  exportCodeAgentContext?: (taskId?: string, targetAgent?: string) => Promise<void>;
  runLocalExperiment?: (taskId?: string) => Promise<void>;
  lastActionTrace?: {
    action: string;
    taskId?: string;
    request?: Record<string, unknown>;
    response?: Record<string, unknown>;
    message: string;
    artifact?: string | null;
    at: string;
  } | null;
};

function tx(locale: Locale | undefined, en: string, zh: string) {
  return locale === "zh-CN" ? zh : en;
}

function parseControlCommand(input: string, taskId: string): ParsedControlCommand {
  const lower = input.toLowerCase();
  if ([
    "patch work order",
    "patch-order",
    "code patch order",
    "repair work order",
    "code-agent patch",
    "code agent patch",
    "turn failure into patch",
    "failure to patch",
    "补丁工单",
    "代码补丁工单",
    "修复工单",
    "生成补丁工单",
    "创建补丁工单",
    "生成代码修复工单",
    "创建代码修复工单",
    "把失败转成补丁",
    "把问题转成补丁",
    "把问题转成工程修复",
    "代码agent修复工单",
    "代码 agent 修复工单"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_patch_work_order",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Turn the latest Scientist failure/blocker evidence into a code-agent patch work order, or block source edits when the issue is an external resource gate."
    };
  }
  if ([
    "upgrade campaign status",
    "candidate campaign status",
    "research parity gate",
    "research parity status",
    "external capability certification",
    "certification status",
    "可信升级活动",
    "研究级对标门禁",
    "外部能力认证"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_upgrade_campaign",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Refresh the external certification, champion campaign, rollback, and research parity gates."
    };
  }
  if ([
    "self-upgrade loop",
    "self upgrade loop",
    "upgrade loop",
    "capability work order",
    "self-upgrade work order",
    "execute self-upgrade",
    "run self-upgrade",
    "自升级闭环",
    "自我升级闭环",
    "能力自升级",
    "生成自升级工单",
    "创建自升级工单",
    "能力缺口转成工单",
    "把 p0 能力缺口转成工单",
    "自进化工程工单"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_self_upgrade_loop",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Select the highest-priority EvoMind capability gap and create a safe code-agent work order with artifacts, gates, acceptance checks, and no training."
    };
  }
  if ([
    "upgrade plan",
    "upgrade backlog",
    "agent upgrade",
    "capability upgrade",
    "engineering upgrade",
    "self upgrade",
    "升级计划",
    "升级 backlog",
    "升级backlog",
    "能力升级",
    "工程升级",
    "自我升级",
    "把 backlog 转成计划",
    "把backlog转成计划"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_upgrade_plan",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Convert the self-audit upgrade backlog into an engineering plan with files, gates, artifacts, and acceptance checks."
    };
  }
  if ([
    "turn plan",
    "tool plan",
    "per-turn plan",
    "plan this turn",
    "plan your tools",
    "what tools will you use",
    "tool rationale",
    "本轮计划",
    "工具计划",
    "行动计划",
    "本次回合",
    "先规划本轮",
    "你准备调用什么工具",
    "你会用哪些工具",
    "每轮计划"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_turn_plan",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Build a read-only per-turn Scientist control plan: intent, tool sequence, rationale, gates, artifacts, and stop conditions."
    };
  }
  if ([
    "readiness report",
    "launch readiness",
    "scientist readiness",
    "agent readiness",
    "go/no-go report",
    "go no-go report",
    "上线报告",
    "上线就绪报告",
    "上线检查报告",
    "最终就绪报告",
    "能力报告",
    "训练就绪报告",
    "能不能上线",
    "能否上线",
    "能不能训练",
    "能否训练",
    "系统是否稳定上线",
    "上线前检查",
    "上线前审计",
    "安全上线检查"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_readiness_report",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Build one read-only EvoMind readiness report: capability, execution gates, claim boundaries, artifacts, and next safe commands."
    };
  }
  if ([
    "causal diagnosis",
    "causal graph",
    "cause map",
    "root cause map",
    "why blocked",
    "why cannot train",
    "diagnose causes",
    "因果诊断",
    "因果图",
    "因果分析",
    "根因图",
    "根因链路",
    "为什么卡住",
    "为什么不能训练",
    "为什么不能上线",
    "为什么不够智能",
    "阻塞归因",
    "问题归因"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_causal_diagnosis",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Build a read-only causal diagnosis graph: symptoms, root causes, evidence, safe interventions, and next command."
    };
  }
  if ([
    "context packet",
    "scientist context",
    "scientist briefing",
    "context briefing",
    "state briefing",
    "research briefing",
    "working context",
    "turn context",
    "生成上下文包",
    "上下文包",
    "科学家上下文",
    "科学家简报",
    "科研简报",
    "状态简报",
    "回合上下文",
    "工作上下文",
    "整理当前上下文"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_context_packet",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Build the per-turn Scientist context packet from task, gates, memory, active strategy, requirement ledger, artifacts, and next safe command."
    };
  }
  if ([
    "strategy optimizer",
    "priority plan",
    "intervention plan",
    "decision matrix",
    "intervention ranking",
    "action ranking",
    "rank interventions",
    "choose next action",
    "best next strategy",
    "下一步策略",
    "策略优化",
    "策略排序",
    "优先级计划",
    "优先级排序",
    "干预排序",
    "行动排序",
    "下一步优先级",
    "先做哪个",
    "应该先做什么",
    "哪个动作最重要",
    "决策矩阵"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_strategy_optimizer",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Rank safe interventions by impact, evidence strength, cost, risk, and gate status before choosing the next command."
    };
  }
  if (["self-audit", "self audit", "capability audit", "agent capability", "intelligence audit", "自我审计", "能力审计", "能力评估", "像 claude code 还差什么", "像 codex 还差什么"].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_self_audit",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Run a read-only EvoMind capability audit: scores, gaps, evidence sources, and upgrade backlog."
    };
  }
  if ([
    "situation model",
    "scientist situation",
    "state model",
    "current situation",
    "research situation",
    "why are we blocked",
    "what should the scientist do next",
    "analyze the current situation",
    "scientist state",
    "局势",
    "情境",
    "态势",
    "当前状态模型",
    "科学家状态",
    "现在局面",
    "现在卡在哪里",
    "为什么卡住",
    "下一步判断",
    "综合证据",
    "综合分析当前"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_situation_model",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Synthesize evidence, uncertainty, blockers, strategy, memory, and the next safe tool sequence without training."
    };
  }
  if ([
    "review hypotheses",
    "hypothesis review",
    "rank hypotheses",
    "critique hypotheses",
    "score hypotheses",
    "proposal review",
    "review proposals",
    "rank proposals",
    "评审假设",
    "假设评审",
    "假设排序",
    "排序假设",
    "评估假设",
    "最佳假设",
    "评审方案",
    "排序方案"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_hypothesis_review",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Review and rank proposed research hypotheses against evidence, readiness, impact, risk, and gates."
    };
  }
  if ([
    "experiment blueprint",
    "candidate blueprint",
    "execution blueprint",
    "plan experiment",
    "gated experiment plan",
    "blueprint",
    "turn hypothesis into experiment",
    "make hypothesis executable"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_experiment_blueprint",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Convert the reviewed hypothesis into a gated, auditable experiment blueprint without starting training."
    };
  }
  if ([
    "innovate-plan",
    "innovation backlog",
    "innovation plan",
    "innovation hypothesis",
    "innovation hypotheses",
    "research hypotheses",
    "memory guided innovation",
    "novel branch",
    "generate innovation",
    "generate hypotheses",
    "创新假设",
    "创新计划",
    "创新分支",
    "生成创新",
    "生成假设",
    "记忆复用",
    "跨任务创新"
  ].some((kw) => lower.includes(kw))) {
    return {
      intent: "scientist_innovation_backlog",
      taskId,
      metadata: { trigger: "ai_control_console", raw_input: input },
      risk: "safe",
      description: "Generate memory-guided research hypotheses and proposal-only branches before training."
    };
  }
  const map: Array<{
    keywords: string[];
    intent: ControlIntent;
    risk: RiskLevel;
    blockedReason?: string;
    description: string;
  }> = [
    {
      keywords: ["contract", "execution contract", "preflight contract", "执行契约", "执行合同", "执行前检查", "运行前检查", "能不能跑", "可以训练吗"],
      intent: "scientist_execution_contract",
      risk: "safe",
      description: "生成执行前契约：go/no-go、AgentSession 就绪、训练就绪、回滚条件和证据清单。"
    },
    {
      keywords: ["workplan", "roadmap", "agenda", "multi-step", "工作计划", "执行计划", "路线图", "多步计划", "持续推进"],
      intent: "scientist_workplan",
      risk: "safe",
      description: "生成可恢复的 AI Scientist 多步工作计划：步骤、门禁、证据、当前焦点和恢复命令。"
    },
    {
      keywords: ["repair", "fix plan", "root cause", "why blocked", "修复计划", "自我修复", "怎么修", "哪里卡住", "阻塞原因", "失败归因"],
      intent: "scientist_repair_plan",
      risk: "safe",
      description: "生成只读自我修复计划：阻塞归因、root cause、修复步骤和 safe next command。"
    },
    {
      keywords: ["resume continuation", "finish remaining safe tools", "auto continue tools", "自动续跑", "自动完成剩余工具", "剩余安全工具自动跑完", "把剩余安全工具跑完", "续跑剩余工具"],
      intent: "scientist_continuation_resume",
      risk: "safe",
      description: "自动续跑上轮未完成的安全只读工具：只调用已门禁的 read-only 工具，遇到训练/提交/停滞立即停止并落证据。"
    },
    {
      keywords: ["continuation status", "continue status", "turn status", "remaining tools", "incomplete tools", "续跑状态", "继续状态", "剩余工具", "还剩哪些工具", "上轮没跑完", "未完成工具"],
      intent: "scientist_continuation_status",
      risk: "safe",
      description: "刷新 AI Scientist 续跑状态：显示已完成工具、剩余安全工具、下一条安全命令和门禁边界。"
    },
    {
      keywords: ["safe next", "next action", "act next", "continue scientist", "安全下一步", "执行安全下一步", "执行下一步", "推进下一步", "继续行动"],
      intent: "scientist_next_action",
      risk: "safe",
      description: "执行安全下一步：只自动运行只读诊断/修复动作；训练、下载、官方提交会停在门禁。"
    },
    {
      keywords: ["recovery", "recover", "resume", "resume context", "checkpoint", "恢复现场", "断点恢复", "上下文丢了", "继续上次", "恢复快照"],
      intent: "scientist_recovery",
      risk: "safe",
      description: "生成长程恢复快照：汇总 guard、turn ledger、step trace、行动队列、阻塞项和恢复命令。"
    },
    {
      keywords: ["loop", "scientist loop", "autonomous loop", "self evolution loop", "自主循环", "持续循环", "自动进化循环", "科学家循环", "自进化循环"],
      intent: "scientist_loop",
      risk: "safe",
      description: "运行有界 AI Scientist 自主循环：诊断、执行只读安全下一步、识别重复、生成修复/契约/计划和经验。"
    },
    {
      keywords: ["autopilot", "diagnose", "diagnosis", "scientist", "全面诊断", "自动诊断", "主动分析", "不够智能", "下一步"],
      intent: "scientist_autopilot",
      risk: "safe",
      description: "运行只读 AI Scientist 诊断链：系统状态、任务、数据、最近实验、记忆、门禁和下一步决策。"
    },
    { keywords: ["onboard", "s6e6", "playground", "接入"], intent: "onboard_playground_s6e6", risk: "gated", description: "通过工作站门禁接入 Playground S6E6。" },
    { keywords: ["create", "workstation", "run", "创建", "新建"], intent: "create_workstation_run", risk: "gated", description: "创建一个新的可审计工作站 run。" },
    { keywords: ["hpc", "gate", "prepare", "算力", "门禁"], intent: "prepare_hpc_execution_gate", risk: "gated", description: "准备 HPC/GPU 执行门禁，不直接启动长训练。" },
    { keywords: ["export", "context", "agent", "导出", "上下文"], intent: "export_code_agent_context", risk: "safe", description: "导出 Code Agent 上下文包，供 Claude Code / Codex 审查。" },
    { keywords: ["deepseek", "draft", "草稿"], intent: "deepseek_code_draft", risk: "gated", description: "生成 DeepSeek Code Agent 草稿，仍需 code review gate。" },
    { keywords: ["claude", "draft", "草稿"], intent: "claude_code_draft", risk: "gated", description: "生成 Claude Code 草稿，仍需 code review gate。" },
    { keywords: ["deepseek", "smoke", "测试"], intent: "deepseek_smoke", risk: "safe", description: "运行 DeepSeek 连接烟测，不打印密钥。" },
    { keywords: ["gpu", "smoke", "连接"], intent: "gpu_smoke", risk: "safe", description: "运行 GPU SSH 网关烟测。" },
    { keywords: ["gpu", "probe"], intent: "gpu_probe_job", risk: "gated", description: "提交 GPU probe job，仅允许 smoke template。" },
    {
      keywords: ["local", "experiment", "本地小任务"],
      intent: "run_local_experiment",
      risk: "gated",
      blockedReason: "本地实验只允许作为工作站资源策略下的受控 smoke；默认训练仍应走 HPC/GPU gate。",
      description: "请求受控小任务实验；若资源策略禁用本地 fallback，后端会返回 blocked artifact。"
    },
    { keywords: ["report", "draft", "报告"], intent: "generate_report_draft", risk: "gated", description: "根据真实 evidence 生成报告草稿。" },
    { keywords: ["evidence", "bundle", "teacher", "证据包"], intent: "generate_teacher_evidence_bundle", risk: "gated", description: "生成教师汇报证据包。" },
    {
      keywords: ["kaggle", "submit", "提交"],
      intent: "kaggle_submit",
      risk: "blocked",
      blockedReason: "官方 Kaggle 提交必须经过 human submission_approval gate，本控制台不会自动提交。",
      description: "官方 Kaggle 提交（被门禁阻断）。"
    }
  ];

  for (const entry of map) {
    if (entry.keywords.some((kw) => lower.includes(kw))) {
      return {
        intent: entry.intent,
        taskId,
        metadata: { trigger: "ai_control_console", raw_input: input },
        risk: entry.risk,
        blockedReason: entry.blockedReason,
        description: entry.description
      };
    }
  }
  return {
    intent: "scientist_turn",
    taskId,
    metadata: { trigger: "ai_control_console", raw_input: input },
    risk: "safe",
    description: "Run one bounded AI Scientist turn: interpret the goal, select safe tools, inspect evidence, write trace artifacts, and stop before training or official submission."
  };
}

function riskTone(risk: RiskLevel): StatusTone {
  return risk === "safe" ? "green" : risk === "gated" ? "amber" : "red";
}

function actionStatusTone(status?: string): StatusTone {
  if (status === "ready") return "blue";
  if (status === "completed") return "green";
  if (status === "blocked" || status === "blocked_until_human_gate") return "red";
  if (status === "pending" || status === "running") return "amber";
  return "slate";
}

function parityPhaseTone(status?: string): StatusTone {
  const normalized = (status ?? "").toLowerCase();
  if (["passed", "complete", "completed", "closed", "ok"].includes(normalized)) return "green";
  if (["planned", "ready"].includes(normalized)) return "blue";
  if (["running", "needs_more_tools", "deferred", "pending"].includes(normalized)) return "amber";
  if (["blocked", "failed", "error"].includes(normalized)) return "red";
  return "slate";
}

function executionGateDecisionTone(status?: string, blocked?: boolean): StatusTone {
  const normalized = (status ?? "").toLowerCase();
  if (blocked || normalized === "blocked") return "red";
  if (normalized === "ready_for_gated_training" || normalized === "ready") return "green";
  if (normalized === "not_run" || normalized === "missing") return "slate";
  if (normalized.includes("go") || normalized.includes("conditional")) return "amber";
  return "slate";
}

function readinessTone(status?: string): StatusTone {
  const normalized = (status ?? "").toLowerCase();
  if (normalized.includes("ready_for_gated_training") || normalized.includes("strong_local_agent_ready")) return "green";
  if (normalized.includes("capability_ready_but_execution_blocked") || normalized.includes("blocked")) return "red";
  if (normalized.includes("needs") || normalized.includes("partial") || normalized.includes("usable")) return "amber";
  if (normalized.includes("not_run") || normalized.includes("unknown")) return "slate";
  return "blue";
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex min-w-0 items-start justify-between gap-3 border-b border-slate-100 py-1.5 text-xs last:border-b-0">
      <span className="shrink-0 font-semibold text-slate-500">{label}</span>
      <span className="min-w-0 break-all text-right font-medium text-slate-800">{value}</span>
    </div>
  );
}

export function AiControlConsole({
  selectedTask,
  locale,
  runWorkstationAction,
  exportCodeAgentContext,
  runLocalExperiment,
  lastActionTrace
}: ScreenProps) {
  const [input, setInput] = useState("");
  const [parsed, setParsed] = useState<ParsedControlCommand | null>(null);
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<ControlMessage[]>([]);
  const [lastResult, setLastResult] = useState<{
    action: string;
    request: string;
    ok: boolean;
    status: number;
    artifact?: string;
    sessionId?: string;
    error?: string;
    rawResponse?: unknown;
  } | null>(null);
  const [autopilot, setAutopilot] = useState<ScientistAutopilotView | null>(null);
  const [scientistActionQueue, setScientistActionQueue] = useState<ScientistActionQueueView | null>(null);
  const [scientistNextAction, setScientistNextAction] = useState<ScientistNextActionView | null>(null);
  const [scientistContinuationStatus, setScientistContinuationStatus] = useState<ScientistContinuationStatusView | null>(null);
  const [scientistContinuationResume, setScientistContinuationResume] = useState<ScientistContinuationResumeView | null>(null);
  const [scientistRecovery, setScientistRecovery] = useState<ScientistRecoveryView | null>(null);
  const [scientistLoop, setScientistLoop] = useState<ScientistLoopView | null>(null);
  const [scientistLoopLessons, setScientistLoopLessons] = useState<ScientistLoopLessonsView | null>(null);
  const [scientistMemoryConsolidation, setScientistMemoryConsolidation] = useState<ScientistMemoryConsolidationView | null>(null);
  const [scientistSelfAudit, setScientistSelfAudit] = useState<ScientistSelfAuditView | null>(null);
  const [scientistReadinessReport, setScientistReadinessReport] = useState<ScientistReadinessReportView | null>(null);
  const [scientistCausalDiagnosis, setScientistCausalDiagnosis] = useState<ScientistCausalDiagnosisView | null>(null);
  const [scientistStrategyOptimizer, setScientistStrategyOptimizer] = useState<ScientistStrategyOptimizerView | null>(null);
  const [scientistContextPacket, setScientistContextPacket] = useState<ScientistContextPacketView | null>(null);
  const [scientistUpgradePlan, setScientistUpgradePlan] = useState<ScientistUpgradePlanView | null>(null);
  const [scientistSelfUpgradeLoop, setScientistSelfUpgradeLoop] = useState<ScientistSelfUpgradeLoopView | null>(null);
  const [scientistUpgradeCampaign, setScientistUpgradeCampaign] = useState<ScientistUpgradeCampaignSummary | null>(null);
  const [scientistPatchWorkOrder, setScientistPatchWorkOrder] = useState<ScientistPatchWorkOrderView | null>(null);
  const [scientistEngineeringLoop, setScientistEngineeringLoop] = useState<ScientistEngineeringLoopView | null>(null);
  const [scientistInnovationBacklog, setScientistInnovationBacklog] = useState<ScientistInnovationBacklogView | null>(null);
  const [scientistHypothesisReview, setScientistHypothesisReview] = useState<ScientistHypothesisReviewView | null>(null);
  const [scientistExperimentBlueprint, setScientistExperimentBlueprint] = useState<ScientistExperimentBlueprintView | null>(null);
  const [scientistSituationModel, setScientistSituationModel] = useState<ScientistSituationModelView | null>(null);
  const [scientistTerminalTurn, setScientistTerminalTurn] = useState<ScientistTerminalTurnView | null>(null);
  const [scientistReasoningSynthesis, setScientistReasoningSynthesis] = useState<ScientistReasoningSynthesisView | null>(null);
  const [scientistTurnPlan, setScientistTurnPlan] = useState<ScientistTurnPlanView | null>(null);
  const [scientistWorkplan, setScientistWorkplan] = useState<ScientistWorkplanView | null>(null);
  const [scientistRepairPlan, setScientistRepairPlan] = useState<ScientistRepairPlanView | null>(null);
  const [scientistExecutionContract, setScientistExecutionContract] = useState<ScientistExecutionContractView | null>(null);
  const [scientistTurns, setScientistTurns] = useState<ScientistTurnsView | null>(null);
  const [scientistStepTrace, setScientistStepTrace] = useState<ScientistStepTraceView | null>(null);
  const [scientistAutopilotStatus, setScientistAutopilotStatus] = useState<ScientistAutopilotStatusView | null>(null);
  const [scientistStream, setScientistStream] = useState<ScientistStreamView | null>(null);
  const [scientistStreamUpdatedAt, setScientistStreamUpdatedAt] = useState<number | null>(null);
  const [scientistStreamTransport, setScientistStreamTransport] = useState<ScientistStreamTransportMode>("connecting");
  const scientistStreamTransportRef = useRef<ScientistStreamTransportMode>("connecting");
  const [autopilotBusy, setAutopilotBusy] = useState(false);
  const [scientistContinuationBusy, setScientistContinuationBusy] = useState(false);
  const [scientistLoopBusy, setScientistLoopBusy] = useState(false);
  const [scientistSelfAuditBusy, setScientistSelfAuditBusy] = useState(false);
  const [scientistReadinessReportBusy, setScientistReadinessReportBusy] = useState(false);
  const [scientistCausalDiagnosisBusy, setScientistCausalDiagnosisBusy] = useState(false);
  const [scientistStrategyOptimizerBusy, setScientistStrategyOptimizerBusy] = useState(false);
  const [scientistContextPacketBusy, setScientistContextPacketBusy] = useState(false);
  const [scientistUpgradePlanBusy, setScientistUpgradePlanBusy] = useState(false);
  const [scientistSelfUpgradeBusy, setScientistSelfUpgradeBusy] = useState(false);
  const [scientistUpgradeCampaignBusy, setScientistUpgradeCampaignBusy] = useState(false);
  const [scientistPromotionApproved, setScientistPromotionApproved] = useState(false);
  const [scientistPatchWorkOrderBusy, setScientistPatchWorkOrderBusy] = useState(false);
  const [scientistEngineeringBusy, setScientistEngineeringBusy] = useState(false);
  const [scientistInnovationBusy, setScientistInnovationBusy] = useState(false);
  const [scientistHypothesisReviewBusy, setScientistHypothesisReviewBusy] = useState(false);
  const [scientistExperimentBlueprintBusy, setScientistExperimentBlueprintBusy] = useState(false);
  const [scientistSituationModelBusy, setScientistSituationModelBusy] = useState(false);
  const [scientistTerminalTurnBusy, setScientistTerminalTurnBusy] = useState(false);
  const [scientistTurnPlanBusy, setScientistTurnPlanBusy] = useState(false);
  const previewRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let alive = true;
    api.getScientistAutopilot()
      .then((payload) => {
        if (alive) setAutopilot(payload.scientist_autopilot);
        if (alive) setScientistActionQueue(payload.scientist_action_queue ?? null);
        if (alive) setScientistWorkplan(payload.scientist_workplan ?? null);
        if (alive) setScientistRepairPlan(payload.scientist_repair_plan ?? null);
        if (alive) setScientistExecutionContract(payload.scientist_execution_contract ?? null);
        if (alive) setScientistTurns(payload.scientist_turns ?? null);
        if (alive) setScientistStepTrace(payload.scientist_step_trace ?? null);
        if (alive) setScientistAutopilotStatus(payload.scientist_autopilot_status ?? null);
      })
      .catch(() => {
        if (alive) setAutopilot(null);
      });
    api.getScientistRecovery()
      .then((payload) => {
        if (alive) setScientistRecovery(payload.scientist_recovery ?? null);
        if (alive && payload.scientist_action_queue) setScientistActionQueue(payload.scientist_action_queue);
        if (alive && payload.scientist_step_trace) setScientistStepTrace(payload.scientist_step_trace);
      })
      .catch(() => {
        if (alive) setScientistRecovery(null);
      });
    api.getScientistLoop()
      .then((payload) => {
        if (alive) setScientistLoop(payload.scientist_loop ?? null);
        if (alive) setScientistLoopLessons(payload.scientist_loop_lessons ?? null);
        if (alive) setScientistMemoryConsolidation(payload.scientist_memory_consolidation ?? null);
        if (alive && payload.scientist_action_queue) setScientistActionQueue(payload.scientist_action_queue);
        if (alive && payload.scientist_step_trace) setScientistStepTrace(payload.scientist_step_trace);
        if (alive && payload.scientist_recovery) setScientistRecovery(payload.scientist_recovery);
      })
      .catch(() => {
        if (alive) setScientistLoop(null);
      });
    api.getScientistContinuationStatus()
      .then((payload) => {
        if (alive) setScientistContinuationStatus(payload.scientist_continuation_status ?? null);
        if (alive && payload.scientist_action_queue) setScientistActionQueue(payload.scientist_action_queue);
        if (alive && payload.scientist_step_trace) setScientistStepTrace(payload.scientist_step_trace);
      })
      .catch(() => {
        if (alive) setScientistContinuationStatus(null);
      });
    api.getScientistContinuationResume()
      .then((payload) => {
        if (alive) setScientistContinuationResume(payload.scientist_continuation_resume ?? null);
        if (alive && payload.scientist_continuation_status) setScientistContinuationStatus(payload.scientist_continuation_status);
        if (alive && payload.scientist_action_queue) setScientistActionQueue(payload.scientist_action_queue);
        if (alive && payload.scientist_step_trace) setScientistStepTrace(payload.scientist_step_trace);
      })
      .catch(() => {
        if (alive) setScientistContinuationResume(null);
      });
    api.getScientistSelfAudit()
      .then((payload) => {
        if (alive) setScientistSelfAudit(payload.scientist_self_audit ?? null);
      })
      .catch(() => {
        if (alive) setScientistSelfAudit(null);
      });
    api.getScientistReadinessReport()
      .then((payload) => {
        if (alive) setScientistReadinessReport(payload.scientist_readiness_report ?? null);
        if (alive && payload.scientist_self_audit) setScientistSelfAudit(payload.scientist_self_audit);
      })
      .catch(() => {
        if (alive) setScientistReadinessReport(null);
      });
    api.getScientistCausalDiagnosis()
      .then((payload) => {
        if (alive) setScientistCausalDiagnosis(payload.scientist_causal_diagnosis ?? null);
        if (alive && payload.scientist_readiness_report) setScientistReadinessReport(payload.scientist_readiness_report);
      })
      .catch(() => {
        if (alive) setScientistCausalDiagnosis(null);
      });
    api.getScientistStrategyOptimizer()
      .then((payload) => {
        if (alive) setScientistStrategyOptimizer(payload.scientist_strategy_optimizer ?? null);
        if (alive && payload.scientist_readiness_report) setScientistReadinessReport(payload.scientist_readiness_report);
        if (alive && payload.scientist_causal_diagnosis) setScientistCausalDiagnosis(payload.scientist_causal_diagnosis);
        if (alive && payload.scientist_action_queue) setScientistActionQueue(payload.scientist_action_queue);
      })
      .catch(() => {
        if (alive) setScientistStrategyOptimizer(null);
      });
    api.getScientistContextPacket()
      .then((payload) => {
        if (alive) setScientistContextPacket(payload.scientist_context_packet ?? null);
        if (alive && payload.scientist_strategy_optimizer) setScientistStrategyOptimizer(payload.scientist_strategy_optimizer);
        if (alive && payload.scientist_readiness_report) setScientistReadinessReport(payload.scientist_readiness_report);
        if (alive && payload.scientist_action_queue) setScientistActionQueue(payload.scientist_action_queue);
      })
      .catch(() => {
        if (alive) setScientistContextPacket(null);
      });
    api.getScientistUpgradePlan()
      .then((payload) => {
        if (alive) setScientistUpgradePlan(payload.scientist_upgrade_plan ?? null);
        if (alive && payload.scientist_self_audit) setScientistSelfAudit(payload.scientist_self_audit);
      })
      .catch(() => {
        if (alive) setScientistUpgradePlan(null);
      });
    api.getScientistSelfUpgradeLoop()
      .then((payload) => {
        if (alive) setScientistSelfUpgradeLoop(payload.scientist_self_upgrade_loop ?? null);
        if (alive && payload.scientist_upgrade_plan) setScientistUpgradePlan(payload.scientist_upgrade_plan);
        if (alive && payload.scientist_self_audit) setScientistSelfAudit(payload.scientist_self_audit);
      })
      .catch(() => {
        if (alive) setScientistSelfUpgradeLoop(null);
      });
    api.getScientistUpgradeCampaign()
      .then((payload) => {
        if (alive) setScientistUpgradeCampaign(payload.scientist_upgrade_campaign ?? null);
      })
      .catch(() => {
        if (alive) setScientistUpgradeCampaign(null);
      });
    api.getScientistPatchWorkOrder()
      .then((payload) => {
        if (alive) setScientistPatchWorkOrder(payload.scientist_patch_work_order ?? null);
        if (alive && payload.scientist_terminal_turn) setScientistTerminalTurn(payload.scientist_terminal_turn);
      })
      .catch(() => {
        if (alive) setScientistPatchWorkOrder(null);
      });
    api.getScientistEngineeringLoop()
      .then((payload) => {
        if (alive) setScientistEngineeringLoop(payload.scientist_engineering_loop ?? null);
      })
      .catch(() => {
        if (alive) setScientistEngineeringLoop(null);
      });
    api.getScientistInnovationBacklog()
      .then((payload) => {
        if (alive) setScientistInnovationBacklog(payload.scientist_innovation_backlog ?? null);
      })
      .catch(() => {
        if (alive) setScientistInnovationBacklog(null);
      });
    api.getScientistHypothesisReview()
      .then((payload) => {
        if (alive) setScientistHypothesisReview(payload.scientist_hypothesis_review ?? null);
      })
      .catch(() => {
        if (alive) setScientistHypothesisReview(null);
      });
    api.getScientistExperimentBlueprint()
      .then((payload) => {
        if (alive) setScientistExperimentBlueprint(payload.scientist_experiment_blueprint ?? null);
      })
      .catch(() => {
        if (alive) setScientistExperimentBlueprint(null);
      });
    api.getScientistSituationModel()
      .then((payload) => {
        if (alive) setScientistSituationModel(payload.scientist_situation_model ?? null);
      })
      .catch(() => {
        if (alive) setScientistSituationModel(null);
      });
    api.getScientistTurnPlan()
      .then((payload) => {
        if (alive) setScientistTurnPlan(payload.scientist_turn_plan ?? null);
      })
      .catch(() => {
        if (alive) setScientistTurnPlan(null);
      });
    api.getScientistTurn()
      .then((payload) => {
        if (alive) setScientistTerminalTurn(payload.scientist_terminal_turn ?? null);
        if (alive) {
          setScientistReasoningSynthesis(
            (payload.scientist_reasoning_synthesis as ScientistReasoningSynthesisView | null | undefined)
            ?? payload.scientist_terminal_turn?.reasoning_synthesis
            ?? null
          );
        }
      })
      .catch(() => {
        if (alive) setScientistTerminalTurn(null);
        if (alive) setScientistReasoningSynthesis(null);
      });
    return () => {
      alive = false;
    };
  }, []);

  function updateScientistStreamTransport(mode: ScientistStreamTransportMode) {
    scientistStreamTransportRef.current = mode;
    setScientistStreamTransport(mode);
  }

  function applyScientistStreamPayload(payload: ScientistStreamPayloadView, transport?: ScientistStreamTransportMode) {
    setScientistStream(payload.scientist_stream ?? null);
    setScientistStreamUpdatedAt(Date.now());
    if (payload.scientist_step_trace) setScientistStepTrace(payload.scientist_step_trace);
    if (payload.scientist_turns) setScientistTurns(payload.scientist_turns);
    const latestTerminalTurn = payload.scientist_terminal_turn
      ?? (payload.scientist_stream?.latest_terminal_turn as ScientistTerminalTurnView | undefined);
    if (latestTerminalTurn) {
      setScientistTerminalTurn(latestTerminalTurn);
      if (latestTerminalTurn.reasoning_synthesis) {
        setScientistReasoningSynthesis(latestTerminalTurn.reasoning_synthesis);
      }
    }
    if (payload.scientist_autopilot_status) setScientistAutopilotStatus(payload.scientist_autopilot_status);
    if (transport) updateScientistStreamTransport(transport);
  }

  function applyScientistStreamEvent(payload: {
    generated_at?: string;
    event_count?: number;
    artifact_path?: string;
    event?: ScientistStreamEventView | null;
  }) {
    const event = payload.event;
    if (!event) return;
    const artifactPath = payload.artifact_path ?? ".xsci/scientist_step_trace.jsonl";
    const eventRecord = event as unknown as Record<string, unknown>;
    setScientistStream((prev) => {
      const recent = [...(prev?.recent_events ?? []), event].slice(-30);
      return {
        ...(prev ?? {}),
        present: true,
        generated_at: payload.generated_at ?? new Date().toISOString(),
        transport: "sse",
        heartbeat: event.ts ?? prev?.heartbeat ?? "",
        artifact_path: artifactPath,
        event_count: payload.event_count ?? prev?.event_count ?? recent.length,
        latest_event: event,
        recent_events: recent,
        no_training_started: true,
        official_submit: "blocked_until_explicit_human_approval",
      };
    });
    setScientistStepTrace((prev) => {
      const recent = [...(prev?.recent ?? []), eventRecord].slice(-80);
      return {
        ...(prev ?? {}),
        present: true,
        artifact_path: artifactPath,
        count: payload.event_count ?? prev?.count ?? recent.length,
        latest: eventRecord,
        recent,
      };
    });
    setScientistStreamUpdatedAt(Date.now());
    updateScientistStreamTransport("sse");
  }

  async function refreshScientistStream() {
    const payload = await api.getScientistStream();
    applyScientistStreamPayload(
      payload,
      scientistStreamTransportRef.current === "sse" ? "sse" : "polling"
    );
  }

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const payload = await api.getScientistStream();
        if (!alive) return;
        applyScientistStreamPayload(
          payload,
          scientistStreamTransportRef.current === "sse" ? "sse" : "polling"
        );
      } catch {
        if (alive) {
          setScientistStream((prev) => prev ?? { present: false, status: "unavailable", recent_events: [] });
          if (scientistStreamTransportRef.current !== "sse") updateScientistStreamTransport("unavailable");
        }
      }
    };
    void tick();
    const timer = window.setInterval(tick, 4000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.EventSource === "undefined") {
      updateScientistStreamTransport("polling");
      return;
    }

    let alive = true;
    const source = new EventSource("/api/scientist/stream/events");
    updateScientistStreamTransport("connecting");

    const parseEventData = <T,>(event: MessageEvent<string>): T | null => {
      try {
        return JSON.parse(event.data) as T;
      } catch {
        return null;
      }
    };

    const handleOpen = () => {
      if (alive) updateScientistStreamTransport("sse");
    };

    const handleSnapshot = (event: MessageEvent<string>) => {
      const payload = parseEventData<ScientistStreamPayloadView>(event);
      if (!alive || !payload) return;
      applyScientistStreamPayload(payload, "sse");
    };

    const handleScientistEvent = (event: MessageEvent<string>) => {
      const payload = parseEventData<{
        generated_at?: string;
        event_count?: number;
        artifact_path?: string;
        event?: ScientistStreamEventView | null;
      }>(event);
      if (!alive || !payload) return;
      applyScientistStreamEvent(payload);
    };

    const handleHeartbeat = () => {
      if (alive) updateScientistStreamTransport("sse");
    };

    source.addEventListener("open", handleOpen);
    source.addEventListener("snapshot", handleSnapshot as EventListener);
    source.addEventListener("scientist_event", handleScientistEvent as EventListener);
    source.addEventListener("heartbeat", handleHeartbeat);
    source.onerror = () => {
      if (!alive) return;
      source.close();
      updateScientistStreamTransport("polling");
    };

    return () => {
      alive = false;
      source.close();
    };
  }, []);

  function pushMessage(role: ControlMessage["role"], content: string) {
    setMessages((prev) => [...prev, { role, content, timestamp: Date.now() }]);
  }

  function parseInput(value = input) {
    if (!value.trim()) return;
    const result = parseControlCommand(value.trim(), selectedTask);
    setParsed(result);
    pushMessage("user", value.trim());
    setTimeout(() => previewRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 30);
  }

  async function executeAction() {
    if (!parsed || parsed.risk === "blocked") return;
    setBusy(true);
    let rawResponse: unknown = null;
    try {
      const taskId = parsed.taskId;
      const meta = parsed.metadata;
      switch (parsed.intent) {
        case "scientist_workplan":
          rawResponse = await runScientistAutopilot();
          pushMessage("system", "Scientist Workplan 已生成：步骤、门禁、证据和当前焦点已更新。");
          break;
        case "scientist_repair_plan":
          rawResponse = await runScientistAutopilot();
          pushMessage("system", "Scientist Repair Plan 已生成：阻塞归因、修复步骤和 safe next command 已更新。");
          break;
        case "scientist_execution_contract":
          rawResponse = await runScientistAutopilot();
          pushMessage("system", "Scientist Execution Contract 已生成：go/no-go、训练就绪、回滚条件和证据清单已更新。");
          break;
        case "scientist_autopilot":
          rawResponse = await runScientistAutopilot();
          pushMessage("system", "Scientist Autopilot 已完成只读诊断。");
          break;
        case "scientist_self_audit":
          rawResponse = await runScientistSelfAudit();
          pushMessage("system", "Scientist Self-Audit completed: capability scores, gaps, and upgrade backlog were refreshed.");
          break;
        case "scientist_readiness_report":
          rawResponse = await runScientistReadinessReport();
          pushMessage("system", "Scientist Readiness Report completed: capability, execution gates, claim boundaries, artifacts, and next safe commands were refreshed. No training was started.");
          break;
        case "scientist_causal_diagnosis":
          rawResponse = await runScientistCausalDiagnosis();
          pushMessage("system", "Scientist Causal Diagnosis completed: symptoms, root causes, evidence, safe interventions, and next command were refreshed. No training was started.");
          break;
        case "scientist_strategy_optimizer":
          rawResponse = await runScientistStrategyOptimizer();
          pushMessage("system", "Scientist Strategy Optimizer completed: safe interventions were ranked by impact, evidence, cost, risk, and gate status. No training was started.");
          break;
        case "scientist_context_packet":
          rawResponse = await runScientistContextPacket();
          pushMessage("system", "Scientist Context Packet completed: task, gates, memory, strategy, requirement ledger, and next safe command were compacted. No training was started.");
          break;
        case "scientist_upgrade_plan":
          rawResponse = await runScientistUpgradePlan();
          pushMessage("system", "Scientist Upgrade Plan completed: open capability backlog was converted into files, gates, artifacts, acceptance checks, and safe next commands. No training was started.");
          break;
        case "scientist_self_upgrade_loop":
          rawResponse = await runScientistSelfUpgradeLoop();
          pushMessage("system", "Scientist Self-Upgrade Loop completed: the next capability gap was converted into a code-agent work order, action queue, trace, and lesson. No training was started.");
          break;
        case "scientist_upgrade_campaign":
          rawResponse = await runScientistUpgradeCampaignAction("status");
          pushMessage("system", "Upgrade campaign status refreshed: external certification and active champion evidence remain independently gated.");
          break;
        case "scientist_patch_work_order":
          rawResponse = await runScientistPatchWorkOrder();
          pushMessage("system", "Scientist Patch Work Order completed: latest failure/blocker evidence was converted into a code-agent work order or safely blocked by an external gate. No training was started.");
          break;
        case "scientist_innovation_backlog":
          rawResponse = await runScientistInnovationBacklog();
          pushMessage("system", "Scientist Innovation Backlog completed: memory-guided hypotheses were generated without training.");
          break;
        case "scientist_hypothesis_review":
          rawResponse = await runScientistHypothesisReview();
          pushMessage("system", "Scientist Hypothesis Review completed: hypotheses were ranked against evidence, readiness, impact, risk, and gates.");
          break;
        case "scientist_experiment_blueprint":
          rawResponse = await runScientistExperimentBlueprint();
          pushMessage("system", "Scientist Experiment Blueprint completed: the reviewed hypothesis is now mapped to branch, code mode, resource mode, artifacts, rollback, and memory writeback gates. No training was started.");
          break;
        case "scientist_situation_model":
          rawResponse = await runScientistSituationModel();
          pushMessage("system", "Scientist Situation Model completed: evidence, uncertainty, blockers, memory, strategy, and next safe tool sequence were refreshed. No training was started.");
          break;
        case "scientist_turn":
          rawResponse = await runScientistTerminalTurn(String(parsed.metadata.raw_input ?? input));
          pushMessage("system", "AI Scientist Turn completed: safe tools, gates, artifacts, execution readiness, and next safe command were refreshed. No training was started.");
          break;
        case "scientist_turn_plan":
          rawResponse = await runScientistTurnPlan();
          pushMessage("system", "Scientist Turn Plan completed: intent, tool sequence, gates, expected artifacts, and stop conditions were refreshed. No training was started.");
          break;
        case "scientist_next_action":
          rawResponse = await executeScientistNextAction();
          break;
        case "scientist_continuation_status":
          rawResponse = await refreshScientistContinuationStatus();
          pushMessage("system", "Scientist Continuation Status 已刷新：剩余安全工具、完成进度和下一条安全命令已更新。");
          break;
        case "scientist_continuation_resume":
          rawResponse = await runScientistContinuationResume();
          pushMessage("system", "Scientist Continuation Resume 已执行：剩余安全只读工具已自动续跑，训练和官方提交仍被门禁阻断。");
          break;
        case "scientist_recovery":
          rawResponse = await runScientistRecovery();
          pushMessage("system", "Scientist Recovery Snapshot 已生成：恢复决策、阻塞项、恢复命令和证据文件已更新。");
          break;
        case "scientist_loop":
          rawResponse = await runScientistLoop();
          pushMessage("system", "Scientist Loop 已完成：循环步骤、停止原因、经验 lesson 和门禁边界已更新。");
          break;
        case "create_workstation_run":
          rawResponse = await runWorkstationAction?.("create_workstation_run", { task_id: taskId, ...meta });
          pushMessage("system", `已为 ${taskId} 创建工作站 run。`);
          break;
        case "onboard_playground_s6e6":
          rawResponse = await runWorkstationAction?.("onboard_playground_s6e6", { task_id: taskId, ...meta });
          pushMessage("system", "S6E6 接入流程已提交到工作站门禁。");
          break;
        case "prepare_hpc_execution_gate":
          rawResponse = await runWorkstationAction?.("prepare_hpc_execution_gate", { task_id: taskId, template: "connection_smoke", ...meta });
          pushMessage("system", "HPC/GPU 执行门禁已准备。");
          break;
        case "export_code_agent_context":
          await exportCodeAgentContext?.(taskId, "claude_code");
          rawResponse = { ok: true, task_id: taskId, target_agent: "claude_code" };
          pushMessage("system", "Code Agent 上下文已导出。");
          break;
        case "deepseek_code_draft":
          rawResponse = await api.generateCodeAgentDraft(taskId, { source_agent: "deepseek_code_agent" });
          pushMessage("system", "DeepSeek Code Agent 草稿已生成。");
          break;
        case "claude_code_draft":
          rawResponse = await api.generateCodeAgentDraft(taskId, { source_agent: "claude_code" });
          pushMessage("system", "Claude Code 草稿已生成。");
          break;
        case "deepseek_smoke":
          rawResponse = await api.testDeepSeek("Hello from EvoMind Gateway");
          pushMessage("system", "DeepSeek 连接烟测完成。");
          break;
        case "gpu_smoke":
          rawResponse = await api.testGpuConnection();
          pushMessage("system", "GPU SSH 网关烟测完成。");
          break;
        case "gpu_probe_job":
          rawResponse = await api.submitGpuJob(taskId, "connection_smoke", meta);
          pushMessage("system", "GPU probe job 已提交，仅使用 smoke template。");
          break;
        case "run_local_experiment":
          await runLocalExperiment?.(taskId);
          rawResponse = { ok: true, task_id: taskId, mode: "workstation_resource_policy_smoke" };
          pushMessage("system", "受控小任务实验请求已交给工作站资源策略处理。");
          break;
        case "generate_report_draft":
          rawResponse = await api.generateReportDraft(taskId, { language: locale ?? "zh-CN", style: "teacher_evidence_bundle" });
          pushMessage("system", "报告草稿已生成。");
          break;
        case "generate_teacher_evidence_bundle":
          rawResponse = await api.generatePaperEvidenceBundle();
          pushMessage("system", "教师汇报证据包已生成。");
          break;
        case "kaggle_submit":
        case "unknown":
          break;
      }

      const data = rawResponse as Record<string, unknown> | null;
      const scientist = data?.scientist_autopilot as ScientistAutopilotView | undefined;
      if (data?.scientist_action_queue) setScientistActionQueue(data.scientist_action_queue as ScientistActionQueueView);
      if (data?.scientist_next_action) setScientistNextAction(data.scientist_next_action as ScientistNextActionView);
      if (data?.scientist_continuation_status) setScientistContinuationStatus(data.scientist_continuation_status as ScientistContinuationStatusView);
      if (data?.scientist_continuation_resume) setScientistContinuationResume(data.scientist_continuation_resume as ScientistContinuationResumeView);
      if (data?.scientist_recovery) setScientistRecovery(data.scientist_recovery as ScientistRecoveryView);
      if (data?.scientist_loop) setScientistLoop(data.scientist_loop as ScientistLoopView);
      if (data?.scientist_loop_lessons) setScientistLoopLessons(data.scientist_loop_lessons as ScientistLoopLessonsView);
      if (data?.scientist_memory_consolidation) setScientistMemoryConsolidation(data.scientist_memory_consolidation as ScientistMemoryConsolidationView);
      if (data?.scientist_self_audit) setScientistSelfAudit(data.scientist_self_audit as ScientistSelfAuditView);
      if (data?.scientist_readiness_report) setScientistReadinessReport(data.scientist_readiness_report as ScientistReadinessReportView);
      if (data?.scientist_causal_diagnosis) setScientistCausalDiagnosis(data.scientist_causal_diagnosis as ScientistCausalDiagnosisView);
      if (data?.scientist_strategy_optimizer) setScientistStrategyOptimizer(data.scientist_strategy_optimizer as ScientistStrategyOptimizerView);
      if (data?.scientist_context_packet) setScientistContextPacket(data.scientist_context_packet as ScientistContextPacketView);
      if (data?.scientist_upgrade_plan) setScientistUpgradePlan(data.scientist_upgrade_plan as ScientistUpgradePlanView);
      if (data?.scientist_self_upgrade_loop) setScientistSelfUpgradeLoop(data.scientist_self_upgrade_loop as ScientistSelfUpgradeLoopView);
      if (data?.scientist_patch_work_order) setScientistPatchWorkOrder(data.scientist_patch_work_order as ScientistPatchWorkOrderView);
      if (data?.scientist_innovation_backlog) setScientistInnovationBacklog(data.scientist_innovation_backlog as ScientistInnovationBacklogView);
      if (data?.scientist_hypothesis_review) setScientistHypothesisReview(data.scientist_hypothesis_review as ScientistHypothesisReviewView);
      if (data?.scientist_experiment_blueprint) setScientistExperimentBlueprint(data.scientist_experiment_blueprint as ScientistExperimentBlueprintView);
      if (data?.scientist_situation_model) setScientistSituationModel(data.scientist_situation_model as ScientistSituationModelView);
      if (data?.scientist_terminal_turn) setScientistTerminalTurn(data.scientist_terminal_turn as ScientistTerminalTurnView);
      if (data?.scientist_engineering_loop) setScientistEngineeringLoop(data.scientist_engineering_loop as ScientistEngineeringLoopView);
      if (data?.scientist_reasoning_synthesis) {
        setScientistReasoningSynthesis(data.scientist_reasoning_synthesis as ScientistReasoningSynthesisView);
      } else if ((data?.scientist_terminal_turn as ScientistTerminalTurnView | undefined)?.reasoning_synthesis) {
        setScientistReasoningSynthesis((data?.scientist_terminal_turn as ScientistTerminalTurnView).reasoning_synthesis ?? null);
      }
      if (data?.scientist_turn_plan) setScientistTurnPlan(data.scientist_turn_plan as ScientistTurnPlanView);
      if (data?.scientist_step_trace) setScientistStepTrace(data.scientist_step_trace as ScientistStepTraceView);
      setLastResult({
        action: parsed.intent,
        request: JSON.stringify(parsed.metadata),
        ok: Boolean(data?.ok ?? true),
        status: 200,
        artifact: (data?.artifact_dir ?? data?.artifact ?? data?.context_dir ?? data?.markdown_path ?? (data?.scientist_context_packet as ScientistContextPacketView | undefined)?.artifact_path ?? (data?.scientist_strategy_optimizer as ScientistStrategyOptimizerView | undefined)?.artifact_path ?? (data?.scientist_causal_diagnosis as ScientistCausalDiagnosisView | undefined)?.artifact_path ?? (data?.scientist_readiness_report as ScientistReadinessReportView | undefined)?.artifact_path ?? (data?.scientist_terminal_turn as ScientistTerminalTurnView | undefined)?.artifact_path ?? (data?.scientist_turn_plan as ScientistTurnPlanView | undefined)?.artifact_path ?? (data?.scientist_continuation_resume as ScientistContinuationResumeView | undefined)?.artifact_path ?? (data?.scientist_continuation_status as ScientistContinuationStatusView | undefined)?.artifact_path ?? (data?.scientist_patch_work_order as ScientistPatchWorkOrderView | undefined)?.artifact_path ?? (data?.scientist_self_upgrade_loop as ScientistSelfUpgradeLoopView | undefined)?.artifact_path ?? (data?.scientist_upgrade_plan as ScientistUpgradePlanView | undefined)?.artifact_path ?? (data?.scientist_situation_model as ScientistSituationModelView | undefined)?.artifact_path ?? (data?.scientist_experiment_blueprint as ScientistExperimentBlueprintView | undefined)?.artifact_path ?? (data?.scientist_hypothesis_review as ScientistHypothesisReviewView | undefined)?.artifact_path ?? (data?.scientist_innovation_backlog as ScientistInnovationBacklogView | undefined)?.artifact_path ?? scientist?.artifact_path) as string | undefined,
        sessionId: (data?.session_id ?? data?.job_id ?? data?.run_id ?? data?.action_id) as string | undefined,
        rawResponse
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      pushMessage("error", `动作失败：${msg}`);
      setLastResult({ action: parsed.intent, request: JSON.stringify(parsed.metadata), ok: false, status: 500, error: msg });
    } finally {
      setBusy(false);
    }
  }

  function quick(label: string, command: string) {
    pushMessage("user", `[Quick Action] ${label}`);
    setInput(command);
    parseInput(command);
  }

  function navigateTo(page: string) {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    url.searchParams.set("page", page);
    window.history.pushState(null, "", url);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }

  async function refreshScientistState() {
    const payload = await api.getScientistAutopilot();
    setAutopilot(payload.scientist_autopilot);
    setScientistActionQueue(payload.scientist_action_queue ?? null);
    setScientistWorkplan(payload.scientist_workplan ?? null);
    setScientistRepairPlan(payload.scientist_repair_plan ?? null);
    setScientistExecutionContract(payload.scientist_execution_contract ?? null);
    setScientistTurns(payload.scientist_turns ?? null);
    setScientistStepTrace(payload.scientist_step_trace ?? null);
    setScientistAutopilotStatus(payload.scientist_autopilot_status ?? null);
    return payload;
  }

  async function refreshScientistRecovery() {
    const payload = await api.getScientistRecovery();
    setScientistRecovery(payload.scientist_recovery ?? null);
    if (payload.scientist_action_queue) setScientistActionQueue(payload.scientist_action_queue);
    if (payload.scientist_step_trace) setScientistStepTrace(payload.scientist_step_trace);
    return payload;
  }

  function wait(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function runScientistAutopilot() {
    setAutopilotBusy(true);
    try {
      const started = await api.startScientistAutopilot();
      setScientistAutopilotStatus(started.scientist_autopilot_status ?? {
        present: true,
        running: true,
        status: "starting",
        run_id: started.run_id,
        pid: started.pid ?? null
      });
      setScientistActionQueue(started.scientist_action_queue ?? null);
      setScientistStepTrace(started.scientist_step_trace ?? null);
      setScientistRepairPlan(started.scientist_repair_plan ?? null);
      setScientistExecutionContract(started.scientist_execution_contract ?? null);
      pushMessage("system", `Scientist Autopilot 已启动：${started.run_id}`);
      let latest = await refreshScientistState();
      for (let i = 0; i < 20; i += 1) {
        const status = latest.scientist_autopilot_status;
        if (status?.running === false && status.status !== "running") break;
        await wait(1000);
        latest = await refreshScientistState();
      }
      await refreshScientistRecovery().catch(() => null);
      return latest;
    } finally {
      setAutopilotBusy(false);
    }
  }

  async function runScientistRecovery() {
    const result = await api.runScientistRecovery();
    setScientistRecovery(result.scientist_recovery ?? null);
    if (result.scientist_action_queue) setScientistActionQueue(result.scientist_action_queue);
    if (result.scientist_step_trace) setScientistStepTrace(result.scientist_step_trace);
    return result;
  }

  async function runScientistLoop() {
    setScientistLoopBusy(true);
    try {
      const result = await api.runScientistLoop();
      setScientistLoop(result.scientist_loop ?? null);
      setScientistLoopLessons(result.scientist_loop_lessons ?? null);
      setScientistMemoryConsolidation(result.scientist_memory_consolidation ?? null);
      if (result.scientist_action_queue) setScientistActionQueue(result.scientist_action_queue);
      if (result.scientist_step_trace) setScientistStepTrace(result.scientist_step_trace);
      if (result.scientist_recovery) setScientistRecovery(result.scientist_recovery);
      return result;
    } finally {
      setScientistLoopBusy(false);
    }
  }

  async function refreshScientistContinuationStatus() {
    setScientistContinuationBusy(true);
    try {
      const result = await api.refreshScientistContinuationStatus();
      setScientistContinuationStatus(result.scientist_continuation_status ?? null);
      if (result.scientist_action_queue) setScientistActionQueue(result.scientist_action_queue);
      if (result.scientist_step_trace) setScientistStepTrace(result.scientist_step_trace);
      return result;
    } finally {
      setScientistContinuationBusy(false);
    }
  }

  async function runScientistContinuationResume() {
    setScientistContinuationBusy(true);
    try {
      const result = await api.runScientistContinuationResume();
      setScientistContinuationResume(result.scientist_continuation_resume ?? null);
      if (result.scientist_continuation_status) setScientistContinuationStatus(result.scientist_continuation_status);
      if (result.scientist_action_queue) setScientistActionQueue(result.scientist_action_queue);
      if (result.scientist_step_trace) setScientistStepTrace(result.scientist_step_trace);
      return result;
    } finally {
      setScientistContinuationBusy(false);
    }
  }

  async function runScientistSelfAudit() {
    setScientistSelfAuditBusy(true);
    try {
      const result = await api.runScientistSelfAudit();
      setScientistSelfAudit(result.scientist_self_audit ?? null);
      return result;
    } finally {
      setScientistSelfAuditBusy(false);
    }
  }

  async function runScientistReadinessReport() {
    setScientistReadinessReportBusy(true);
    try {
      const result = await api.runScientistReadinessReport();
      setScientistReadinessReport(result.scientist_readiness_report ?? null);
      if (result.scientist_self_audit) setScientistSelfAudit(result.scientist_self_audit);
      return result;
    } finally {
      setScientistReadinessReportBusy(false);
    }
  }

  async function runScientistCausalDiagnosis() {
    setScientistCausalDiagnosisBusy(true);
    try {
      const result = await api.runScientistCausalDiagnosis();
      setScientistCausalDiagnosis(result.scientist_causal_diagnosis ?? null);
      if (result.scientist_readiness_report) setScientistReadinessReport(result.scientist_readiness_report);
      return result;
    } finally {
      setScientistCausalDiagnosisBusy(false);
    }
  }

  async function runScientistStrategyOptimizer() {
    setScientistStrategyOptimizerBusy(true);
    try {
      const result = await api.runScientistStrategyOptimizer();
      setScientistStrategyOptimizer(result.scientist_strategy_optimizer ?? null);
      if (result.scientist_readiness_report) setScientistReadinessReport(result.scientist_readiness_report);
      if (result.scientist_causal_diagnosis) setScientistCausalDiagnosis(result.scientist_causal_diagnosis);
      if (result.scientist_action_queue) setScientistActionQueue(result.scientist_action_queue);
      return result;
    } finally {
      setScientistStrategyOptimizerBusy(false);
    }
  }

  async function runScientistContextPacket() {
    setScientistContextPacketBusy(true);
    try {
      const result = await api.runScientistContextPacket();
      setScientistContextPacket(result.scientist_context_packet ?? null);
      if (result.scientist_strategy_optimizer) setScientistStrategyOptimizer(result.scientist_strategy_optimizer);
      if (result.scientist_readiness_report) setScientistReadinessReport(result.scientist_readiness_report);
      if (result.scientist_action_queue) setScientistActionQueue(result.scientist_action_queue);
      return result;
    } finally {
      setScientistContextPacketBusy(false);
    }
  }

  async function runScientistUpgradePlan() {
    setScientistUpgradePlanBusy(true);
    try {
      const result = await api.runScientistUpgradePlan();
      setScientistUpgradePlan(result.scientist_upgrade_plan ?? null);
      if (result.scientist_self_audit) setScientistSelfAudit(result.scientist_self_audit);
      return result;
    } finally {
      setScientistUpgradePlanBusy(false);
    }
  }

  async function runScientistSelfUpgradeLoop() {
    setScientistSelfUpgradeBusy(true);
    try {
      const result = await api.runScientistSelfUpgradeLoop();
      setScientistSelfUpgradeLoop(result.scientist_self_upgrade_loop ?? null);
      if (result.scientist_upgrade_plan) setScientistUpgradePlan(result.scientist_upgrade_plan);
      if (result.scientist_self_audit) setScientistSelfAudit(result.scientist_self_audit);
      return result;
    } finally {
      setScientistSelfUpgradeBusy(false);
    }
  }

  async function runScientistUpgradeCampaignAction(action: "status" | "run" | "promote" | "rollback") {
    setScientistUpgradeCampaignBusy(true);
    try {
      if (action !== "status") {
        await api.runScientistUpgradeCampaign({
          action,
          human_approved: action === "promote" ? scientistPromotionApproved : undefined,
          timeout_seconds: 300
        });
      }
      const refreshed = await api.getScientistUpgradeCampaign();
      setScientistUpgradeCampaign(refreshed.scientist_upgrade_campaign ?? null);
      if (action === "promote") setScientistPromotionApproved(false);
      return refreshed;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upgrade campaign action failed";
      pushMessage("error", message);
      return null;
    } finally {
      setScientistUpgradeCampaignBusy(false);
    }
  }

  async function runScientistPatchWorkOrder() {
    setScientistPatchWorkOrderBusy(true);
    try {
      const result = await api.runScientistPatchWorkOrder();
      setScientistPatchWorkOrder(result.scientist_patch_work_order ?? null);
      if (result.scientist_terminal_turn) setScientistTerminalTurn(result.scientist_terminal_turn);
      await refreshScientistStream().catch(() => null);
      return result;
    } finally {
      setScientistPatchWorkOrderBusy(false);
    }
  }

  async function runScientistEngineeringLoop(generatePatch: boolean) {
    setScientistEngineeringBusy(true);
    try {
      const result = await api.runScientistEngineeringLoop({
        generatePatch,
        timeoutSeconds: 300
      });
      setScientistEngineeringLoop(result.scientist_engineering_loop ?? null);
      await refreshScientistStream().catch(() => null);
      return result;
    } finally {
      setScientistEngineeringBusy(false);
    }
  }

  async function runScientistInnovationBacklog() {
    setScientistInnovationBusy(true);
    try {
      const result = await api.runScientistInnovationBacklog();
      setScientistInnovationBacklog(result.scientist_innovation_backlog ?? null);
      if (result.scientist_self_audit) setScientistSelfAudit(result.scientist_self_audit);
      return result;
    } finally {
      setScientistInnovationBusy(false);
    }
  }

  async function runScientistHypothesisReview() {
    setScientistHypothesisReviewBusy(true);
    try {
      const result = await api.runScientistHypothesisReview();
      setScientistHypothesisReview(result.scientist_hypothesis_review ?? null);
      return result;
    } finally {
      setScientistHypothesisReviewBusy(false);
    }
  }

  async function runScientistExperimentBlueprint() {
    setScientistExperimentBlueprintBusy(true);
    try {
      const result = await api.runScientistExperimentBlueprint();
      setScientistExperimentBlueprint(result.scientist_experiment_blueprint ?? null);
      return result;
    } finally {
      setScientistExperimentBlueprintBusy(false);
    }
  }

  async function runScientistSituationModel() {
    setScientistSituationModelBusy(true);
    try {
      const result = await api.runScientistSituationModel();
      setScientistSituationModel(result.scientist_situation_model ?? null);
      return result;
    } finally {
      setScientistSituationModelBusy(false);
    }
  }

  async function runScientistTerminalTurn(prompt: string) {
    setScientistTerminalTurnBusy(true);
    try {
      const result = await api.runScientistTurn(
        prompt || "Analyze the current EvoMind workstation state and propose the next safe research step.",
        4
      );
      setScientistTerminalTurn(result.scientist_terminal_turn ?? null);
      setScientistReasoningSynthesis(
        (result.scientist_reasoning_synthesis as ScientistReasoningSynthesisView | null | undefined)
        ?? result.scientist_terminal_turn?.reasoning_synthesis
        ?? null
      );
      if (result.scientist_context_packet) setScientistContextPacket(result.scientist_context_packet);
      if (result.scientist_strategy_optimizer) setScientistStrategyOptimizer(result.scientist_strategy_optimizer);
      if (result.scientist_action_queue) setScientistActionQueue(result.scientist_action_queue);
      if (result.scientist_step_trace) setScientistStepTrace(result.scientist_step_trace);
      await refreshScientistStream().catch(() => null);
      return result;
    } finally {
      setScientistTerminalTurnBusy(false);
    }
  }

  async function runScientistTurnPlan() {
    setScientistTurnPlanBusy(true);
    try {
      const result = await api.runScientistTurnPlan();
      setScientistTurnPlan(result.scientist_turn_plan ?? null);
      return result;
    } finally {
      setScientistTurnPlanBusy(false);
    }
  }

  async function executeScientistNextAction() {
    setBusy(true);
    try {
      const result = await api.runScientistNextAction();
      setScientistNextAction(result.scientist_next_action ?? null);
      setScientistActionQueue(result.scientist_action_queue ?? null);
      setScientistStepTrace(result.scientist_step_trace ?? null);
      const status = result.scientist_next_action?.status ?? "unknown";
      const selected = result.scientist_next_action?.selected_action?.id ?? "none";
      pushMessage("system", `Safe Next 已处理：status=${status}, selected=${selected}。训练/提交保持门禁。`);
      await refreshScientistContinuationStatus().catch(() => null);
      return result;
    } finally {
      setBusy(false);
    }
  }

  const quickActions = [
    { label: tx(locale, "Scientist Turn", "科学家回合"), icon: Send, command: "分析当前研究状态，调用安全工具，给出下一步可执行建议" },
    { label: tx(locale, "Turn Plan", "本轮计划"), icon: BrainCircuit, command: "turn plan tool rationale" },
    { label: tx(locale, "Scientist Workplan", "科学家工作计划"), icon: GitBranch, command: "scientist workplan roadmap" },
    { label: tx(locale, "Repair Plan", "修复计划"), icon: ShieldCheck, command: "scientist repair root cause" },
    { label: tx(locale, "Execution Contract", "执行契约"), icon: FileCheck2, command: "scientist execution contract preflight" },
    { label: tx(locale, "Situation Model", "局势模型"), icon: BrainCircuit, command: "situation model current research state" },
    { label: tx(locale, "Scientist Autopilot", "科学家诊断"), icon: BrainCircuit, command: "scientist autopilot diagnose" },
    { label: tx(locale, "Self Audit", "自我审计"), icon: ShieldCheck, command: "self-audit capability audit" },
    { label: tx(locale, "Readiness Report", "就绪报告"), icon: FileCheck2, command: "readiness report launch gates" },
    { label: tx(locale, "Causal Diagnosis", "因果诊断"), icon: GitBranch, command: "causal diagnosis root cause map" },
    { label: tx(locale, "Strategy Optimizer", "策略优化"), icon: GitBranch, command: "strategy optimizer intervention ranking" },
    { label: tx(locale, "Context Packet", "上下文包"), icon: BrainCircuit, command: "scientist context packet briefing" },
    { label: tx(locale, "Upgrade Plan", "升级计划"), icon: GitBranch, command: "upgrade plan from self-audit backlog" },
    { label: tx(locale, "Patch Order", "补丁工单"), icon: ShieldCheck, command: "把失败转成代码补丁工单" },
    { label: tx(locale, "Research Hypotheses", "创新假设"), icon: Lightbulb, command: "innovation backlog research hypotheses" },
    { label: tx(locale, "Hypothesis Review", "假设评审"), icon: FileCheck2, command: "review hypotheses rank proposals" },
    { label: tx(locale, "Experiment Blueprint", "实验蓝图"), icon: GitBranch, command: "experiment blueprint gated plan" },
    { label: tx(locale, "Safe Next", "安全下一步"), icon: Play, command: "execute safe next action" },
    { label: tx(locale, "Continuation Status", "续跑状态"), icon: History, command: "continuation status remaining tools" },
    { label: tx(locale, "Recovery Snapshot", "恢复现场"), icon: History, command: "scientist recovery snapshot" },
    { label: tx(locale, "Scientist Loop", "科学家自主循环"), icon: RefreshCcw, command: "scientist autonomous loop" },
    { label: tx(locale, "Create Run", "创建 Run"), icon: GitBranch, command: "create workstation run" },
    { label: tx(locale, "Export Context", "导出上下文"), icon: TerminalSquare, command: "export code agent context" },
    { label: tx(locale, "DeepSeek Draft", "DeepSeek 草稿"), icon: BrainCircuit, command: "deepseek draft" },
    { label: tx(locale, "Claude Draft", "Claude 草稿"), icon: Bot, command: "claude code draft" },
    { label: tx(locale, "DeepSeek Smoke", "DeepSeek 烟测"), icon: BrainCircuit, command: "deepseek smoke test" },
    { label: tx(locale, "GPU Smoke", "GPU 烟测"), icon: Cpu, command: "gpu smoke test" },
    { label: tx(locale, "HPC Gate", "HPC 门禁"), icon: ShieldCheck, command: "prepare hpc execution gate" },
    { label: tx(locale, "Report Draft", "报告草稿"), icon: FileCheck2, command: "generate report draft" },
    { label: tx(locale, "Evidence Bundle", "证据包"), icon: Upload, command: "generate teacher evidence bundle" }
  ];

  const loopFinalNext = (scientistLoop?.final_next_action && typeof scientistLoop.final_next_action === "object"
    ? scientistLoop.final_next_action as Record<string, unknown>
    : null);
  const loopFinalSelected = (loopFinalNext?.selected_action && typeof loopFinalNext.selected_action === "object"
    ? loopFinalNext.selected_action as Record<string, unknown>
    : null);
  const loopLesson = (scientistLoop?.lesson && typeof scientistLoop.lesson === "object"
    ? scientistLoop.lesson as Record<string, unknown>
    : null);
  const latestStoredLesson = (scientistLoopLessons?.latest && typeof scientistLoopLessons.latest === "object"
    ? scientistLoopLessons.latest as Record<string, unknown>
    : null);
  const continuationRemaining = scientistContinuationStatus?.remaining_safe_tools ?? [];
  const continuationCompleted = scientistContinuationStatus?.executed_or_completed_tools ?? [];
  const continuationHistory = scientistContinuationStatus?.progress_history ?? [];
  const continuationCompletedCount = scientistContinuationStatus?.completed_required_tools ?? continuationCompleted.length;
  const continuationTotal = scientistContinuationStatus?.total_required_tools ?? (continuationCompletedCount + continuationRemaining.length);
  const continuationRatio = scientistContinuationStatus?.completion_ratio ?? (continuationTotal > 0 ? continuationCompletedCount / continuationTotal : 0);
  const continuationNextCommand = scientistContinuationStatus?.next_safe_action_command ?? scientistContinuationStatus?.safe_next_command ?? "evomind turn";
  const situationBody = scientistSituationModel?.situation_model ?? null;
  const situationReadiness = scientistSituationModel?.readiness_score ?? situationBody?.readiness_score ?? 0;
  const situationChecks = situationBody?.readiness_checks ?? {};
  const situationMissingChecks = Object.entries(situationChecks).filter(([, value]) => !value).map(([key]) => key);
  const situationPassedChecks = Object.entries(situationChecks).filter(([, value]) => value).map(([key]) => key);
  const situationBlockers = situationBody?.blocker_model ?? [];
  const situationUncertainties = situationBody?.uncertainties ?? [];
  const situationNextCommands = scientistSituationModel?.next_safe_commands ?? situationBody?.recommended_tool_sequence ?? [];
  const turnPlanIntent = scientistTurnPlan?.intent?.kind ?? "not_run";
  const turnPlanPayload = scientistTurnPlan?.intent?.payload ?? "";
  const turnPlanReadiness = scientistTurnPlan?.readiness ?? {};
  const turnPlanBlockingGates = Array.isArray(turnPlanReadiness.blocking_gates) ? turnPlanReadiness.blocking_gates.map(String) : [];
  const turnPlanAdvisoryGaps = Array.isArray(turnPlanReadiness.advisory_gaps) ? turnPlanReadiness.advisory_gaps.map(String) : [];
  const turnPlanTools = scientistTurnPlan?.selected_tools ?? [];
  const turnPlanStopConditions = scientistTurnPlan?.stop_conditions ?? [];
  const turnPlanExpectedArtifacts = scientistTurnPlan?.expected_artifacts ?? [];
  const turnPlanCritique = scientistTurnPlan?.scientific_critique ?? scientistTerminalTurn?.scientific_critique ?? null;
  const scientistRequirementLedger = scientistTerminalTurn?.requirement_ledger ?? scientistTurnPlan?.requirement_ledger ?? null;
  const scientistOpenRequirements = scientistRequirementLedger?.open_requirements ?? [];
  const scientistBlockedRequirements = scientistRequirementLedger?.blocked_requirements ?? [];
  const scientistNextEvidence = scientistRequirementLedger?.next_evidence_to_collect ?? [];
  const scientistToolBudget = scientistTerminalTurn?.tool_budget ?? scientistTurnPlan?.tool_budget ?? null;
  const scientistDeferredTools = scientistTerminalTurn?.deferred_tools ?? [];
  const scientistMustRunDeferredTools = scientistTerminalTurn?.must_run_deferred_tools ?? [];
  const scientistBudgetExhausted = Boolean(scientistTerminalTurn?.budget_exhausted);
  const turnPlanEvidenceGaps = Array.isArray(turnPlanCritique?.evidence_gaps) ? turnPlanCritique.evidence_gaps : [];
  const turnPlanClaimBoundaries = Array.isArray(turnPlanCritique?.claim_boundaries) ? turnPlanCritique.claim_boundaries : [];
  const turnPlanUncertainty = Array.isArray(turnPlanCritique?.uncertainty_drivers) ? turnPlanCritique.uncertainty_drivers : [];
  const scientistParityLifecycle = scientistTerminalTurn?.parity_lifecycle ?? scientistTurnPlan?.parity_lifecycle ?? null;
  const scientistParityPhases: ScientistParityPhaseView[] = Array.isArray(scientistParityLifecycle?.phases) && scientistParityLifecycle.phases.length
    ? scientistParityLifecycle.phases
    : (["observe", "plan", "act", "reflect", "improve"] as const).map((phase) => ({
      phase,
      status: "not_run",
      purpose: tx(
        locale,
        "Waiting for an AI Scientist turn to record this phase.",
        "等待 AI Scientist 回合记录该阶段。"
      ),
    }));
  const scientistParityArtifact = scientistTerminalTurn?.parity_loop_artifact
    ?? String(scientistParityLifecycle?.completion_gate?.must_record_artifact ?? ".xsci/scientist_parity_loop.jsonl");
  const scientistParityPassedCount = scientistParityPhases.filter((phase) => parityPhaseTone(phase.status) === "green").length;
  const scientistParityGateTone: StatusTone = scientistParityLifecycle?.budget_exhausted
    ? "amber"
    : scientistParityLifecycle
      ? "green"
      : "slate";
  const streamEvents = scientistStream?.recent_events ?? [];
  const latestStreamEvent = scientistStream?.latest_event ?? null;
  const streamStatus = scientistStream?.running ? "running" : (scientistStream?.status ?? "idle");
  const streamTone: StatusTone = scientistStream?.running
    ? "amber"
    : streamStatus === "failed" || streamStatus === "blocked"
      ? "red"
      : scientistStream?.present
        ? "green"
        : "slate";
  const streamLastUpdated = scientistStreamUpdatedAt ? new Date(scientistStreamUpdatedAt).toLocaleTimeString() : "-";
  const streamTransportTone: StatusTone = scientistStreamTransport === "sse"
    ? "green"
    : scientistStreamTransport === "connecting"
      ? "amber"
      : scientistStreamTransport === "unavailable"
        ? "red"
        : "slate";
  const streamTransportLabel = scientistStreamTransport === "sse" ? "live" : scientistStreamTransport;
  const executionGateDecision = scientistExecutionContract?.execution_gate_decision ?? null;
  const executionGateStatus = executionGateDecision?.status ?? (scientistExecutionContract?.present ? "legacy_contract" : "not_run");
  const executionGateTone = executionGateDecisionTone(executionGateStatus, executionGateDecision?.blocked);
  const executionGateBlockedBy = executionGateDecision?.blocked_by ?? [];
  const executionGateSafeCommands = executionGateDecision?.safe_next_commands ?? [];
  const executionGateSetupBlockers = executionGateDecision?.setup_blockers ?? scientistExecutionContract?.setup_blockers ?? [];
  const executionGateRootCauses = executionGateDecision?.root_causes ?? scientistExecutionContract?.root_causes ?? [];
  const executionGateNoTrainingStarted = executionGateDecision?.no_training_started ?? scientistExecutionContract?.no_training_started ?? true;
  const executionGateOfficialSubmit = executionGateDecision?.official_submit ?? scientistExecutionContract?.official_submit ?? "blocked_until_explicit_human_approval";
  const terminalTurnTools = scientistTerminalTurn?.executed_tools ?? [];
  const terminalTurnBlockers = scientistTerminalTurn?.blocking_gates ?? [];
  const terminalTurnArtifacts = scientistTerminalTurn?.artifacts ?? [];
  const reasoningSynthesis = scientistReasoningSynthesis ?? scientistTerminalTurn?.reasoning_synthesis ?? null;
  const reasoningHypotheses = reasoningSynthesis?.hypotheses ?? [];
  const reasoningQuality = reasoningSynthesis?.reasoning_quality;
  const reasoningNextAction = reasoningSynthesis?.next_safe_action;
  const terminalTurnStatus = scientistTerminalTurn?.execution_blocked
    ? "blocked"
    : scientistTerminalTurn?.execution_ready
      ? "ready"
      : scientistTerminalTurn?.present
        ? "observed"
        : "not_run";
  const terminalTurnTone: StatusTone = terminalTurnStatus === "ready"
    ? "green"
    : terminalTurnStatus === "blocked"
      ? "red"
      : scientistTerminalTurn?.present
        ? "blue"
        : "slate";
  const scientistPatchOrderBody = scientistPatchWorkOrder?.work_order ?? null;
  const scientistPatchActions = Array.isArray((scientistPatchWorkOrder?.action_queue as ScientistPatchActionQueueView | undefined)?.actions)
    ? ((scientistPatchWorkOrder?.action_queue as ScientistPatchActionQueueView).actions ?? [])
    : [];
  const scientistPatchStatus = scientistPatchWorkOrder?.status ?? scientistPatchOrderBody?.status ?? "not_run";
  const scientistPatchTone: StatusTone = scientistPatchStatus === "ready_for_code_agent"
    ? "green"
    : scientistPatchStatus === "blocked_external_gate"
      ? "red"
      : scientistPatchWorkOrder?.present
        ? "amber"
        : "slate";
  const scientistEngineeringStatus = scientistEngineeringLoop?.status ?? "not_run";
  const scientistEngineeringChecks = scientistEngineeringLoop?.acceptance_checks ?? [];
  const scientistEngineeringPassed = scientistEngineeringChecks.filter((item) => item.passed).length;
  const scientistEngineeringTone: StatusTone = scientistEngineeringLoop?.merge_ready
    ? "green"
    : scientistEngineeringStatus.startsWith("blocked") || scientistEngineeringStatus.startsWith("failed")
      ? "red"
      : scientistEngineeringLoop?.present
        ? "amber"
        : "slate";
  const scientistCertificationEvidence = (scientistUpgradeCampaign?.certification ?? {}) as Record<string, unknown>;
  const scientistCampaignEvidence = (scientistUpgradeCampaign?.upgrade_campaign ?? {}) as Record<string, unknown>;
  const scientistCampaignStatus = String(
    scientistCampaignEvidence.campaign_status
      ?? scientistCampaignEvidence.status
      ?? scientistUpgradeCampaign?.status
      ?? "not_run"
  );
  const scientistCertificationStatus = String(scientistCertificationEvidence.status ?? "not_certified");
  const scientistParityCertified = scientistUpgradeCampaign?.parity_claim_allowed === true;
  const scientistCampaignTone: StatusTone = scientistParityCertified ? "green" : "red";

  return (
    <main className="space-y-4">
      <div>
        <h2 className="text-xl font-bold tracking-normal text-slate-950">{tx(locale, "EvoMind Gateway", "EvoMind 工作站入口")}</h2>
        <p className="mt-1 text-sm leading-6 text-slate-500">
          {tx(locale, "Command the workstation with natural language. All actions are gated and logged.", "用自然语言调度工作站；所有动作都经过门禁并写入审计日志。")}
        </p>
      </div>

      <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs leading-5 text-amber-800">
        <strong>{tx(locale, "Safety Rules:", "安全规则：")}</strong>{" "}
        {tx(
          locale,
          "Code Agents never bypass the workstation. Training requires a workstation action or GPU job manifest. Official Kaggle submission requires human approval. Secrets are never displayed.",
          "Code Agent 不绕过工作站；训练必须由 workstation action 或 GPU job manifest 发起；官方 Kaggle 提交必须有人类审批；页面不展示任何密钥。"
        )}
      </div>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "AI Scientist Turn", "AI Scientist 回合")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Run one bounded research turn: interpret the goal, choose safe tools, inspect evidence, write trace artifacts, and stop before training or official submission.",
                "执行一次受控科研回合：理解目标、选择安全工具、检查证据、写入轨迹文件，并在训练或官方提交前停止。"
              )}
            </CardDescription>
          </div>
          <Button
            size="sm"
            variant="secondary"
            onClick={() => void runScientistTerminalTurn(input.trim())}
            disabled={scientistTerminalTurnBusy}
          >
            {scientistTerminalTurnBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            {tx(locale, "Run Scientist Turn", "运行科学家回合")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 xl:grid-cols-[0.72fr_1.28fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Status", "状态")} value={<StatusBadge tone={terminalTurnTone}>{terminalTurnStatus}</StatusBadge>} />
            <Row label={tx(locale, "Task", "任务")} value={scientistTerminalTurn?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Autonomy", "自主级别")} value={scientistTerminalTurn?.autonomy_level ?? "not_run"} />
            <Row label={tx(locale, "Execution Ready", "可执行")} value={String(scientistTerminalTurn?.execution_ready ?? false)} />
            <Row label={tx(locale, "Tools", "工具数")} value={terminalTurnTools.length} />
            <Row
              label={tx(locale, "Tool Budget", "工具预算")}
              value={`${scientistToolBudget?.executed_tool_count ?? terminalTurnTools.length}/${scientistToolBudget?.effective_max_tools ?? scientistToolBudget?.recommended_min_tools ?? "-"}`}
            />
            <Row
              label={tx(locale, "Budget Gate", "预算门禁")}
              value={<StatusBadge tone={scientistBudgetExhausted ? "amber" : scientistTerminalTurn?.present ? "green" : "slate"}>{scientistBudgetExhausted ? "deferred" : scientistTerminalTurn?.present ? "complete" : "not_run"}</StatusBadge>}
            />
            <Row label={tx(locale, "Next Safe", "安全下一步")} value={scientistTerminalTurn?.next_safe_command ?? "evomind ask \"analyze current state\""} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistTerminalTurn?.artifact_path ?? ".xsci/scientist_terminal_turn.json"} />
          </div>
          <div className="grid gap-3 lg:grid-cols-3">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Goal", "目标")}</div>
              <div className="thin-scrollbar max-h-40 overflow-y-auto text-xs leading-5 text-slate-700">
                {scientistTerminalTurn?.user_goal || tx(locale, "No terminal turn has been run yet.", "尚未运行终端科学家回合。")}
              </div>
              <div className="mt-3 text-[11px] font-semibold text-slate-500">
                {tx(locale, "Official submit", "官方提交")} = {scientistTerminalTurn?.official_submit ?? "blocked_until_explicit_human_approval"}
              </div>
            </div>
            <div className="thin-scrollbar max-h-56 overflow-y-auto rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Executed Tools", "已执行工具")}</div>
              {scientistToolBudget ? (
                <div className="mb-2 rounded border border-indigo-100 bg-indigo-50 px-2 py-1.5 text-[11px] leading-4 text-indigo-800">
                  <div>
                    {tx(locale, "recommended", "推荐")}={scientistToolBudget.recommended_min_tools ?? "-"} · {tx(locale, "effective", "实际")}={scientistToolBudget.effective_max_tools ?? "-"} · {tx(locale, "requested", "请求")}={scientistToolBudget.requested_max_tools ?? "-"}
                  </div>
                  {scientistToolBudget.expansion_reason ? <div className="mt-1">{scientistToolBudget.expansion_reason}</div> : null}
                </div>
              ) : null}
              {terminalTurnTools.length === 0 ? (
                <div className="text-xs text-slate-400">{tx(locale, "No tools have run yet.", "尚未执行工具。")}</div>
              ) : null}
              <div className="space-y-2">
                {terminalTurnTools.slice(0, 10).map((item, index) => (
                  <div key={`${item.tool ?? "tool"}-${index}`} className="rounded border border-blue-100 bg-blue-50 px-2 py-1.5 text-xs">
                    <div className="flex items-center justify-between gap-2">
                      <span className="break-all font-mono font-bold text-blue-900">{item.tool ?? "unknown_tool"}</span>
                      <StatusBadge tone={item.ok === false ? "red" : "green"}>{item.ok === false ? "blocked" : "ok"}</StatusBadge>
                    </div>
                    {item.message ? <div className="mt-1 leading-4 text-blue-800">{item.message}</div> : null}
                    {item.artifact_path ? <div className="mt-1 break-all font-mono text-[11px] text-blue-700">{item.artifact_path}</div> : null}
                  </div>
                ))}
              </div>
              {scientistDeferredTools.length ? (
                <div className="mt-2 rounded border border-amber-100 bg-amber-50 p-2 text-xs leading-4 text-amber-800">
                  <div className="mb-1 font-bold">
                    {tx(locale, "Deferred Tools", "延期工具")}
                    {scientistMustRunDeferredTools.length ? ` · ${tx(locale, "must-run pending", "关键待执行")}` : ""}
                  </div>
                  {scientistDeferredTools.slice(0, 6).map((tool, index) => (
                    <div key={`${tool}-${index}`} className="break-all font-mono text-[11px]">
                      {scientistMustRunDeferredTools.includes(tool) ? "! " : ""}
                      {tool}
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
            <div className="thin-scrollbar max-h-56 overflow-y-auto rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Gates & Artifacts", "门禁与证据")}</div>
              {terminalTurnBlockers.length ? (
                <div className="space-y-2">
                  {terminalTurnBlockers.slice(0, 4).map((item, index) => (
                    <div key={`${item}-${index}`} className="rounded border border-red-100 bg-red-50 px-2 py-1.5 text-xs leading-4 text-red-800">
                      {item}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded border border-green-100 bg-green-50 px-2 py-1.5 text-xs leading-4 text-green-800">
                  {tx(locale, "No blocking gate recorded for the last Scientist Turn.", "最近一次科学家回合没有记录阻断门禁。")}
                </div>
              )}
              <div className="mt-2 space-y-1">
                {terminalTurnArtifacts.slice(0, 6).map((item, index) => (
                  <div key={`${item}-${index}`} className="break-all rounded bg-slate-50 px-2 py-1 font-mono text-[11px] text-slate-700">
                    {item}
                  </div>
                ))}
              </div>
            </div>
            <div className="thin-scrollbar max-h-56 overflow-y-auto rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-xs font-bold uppercase text-slate-500">{tx(locale, "Requirements", "需求闭环")}</div>
                <StatusBadge tone={scientistBlockedRequirements.length ? "red" : scientistOpenRequirements.length ? "amber" : scientistRequirementLedger ? "green" : "slate"}>
                  {scientistRequirementLedger ? `${scientistOpenRequirements.length} open` : "not_run"}
                </StatusBadge>
              </div>
              {scientistRequirementLedger ? (
                <div className="space-y-2 text-xs leading-4 text-slate-700">
                  {scientistBlockedRequirements.length ? (
                    <div className="rounded border border-red-100 bg-red-50 p-2 text-red-800">
                      <div className="mb-1 font-bold">{tx(locale, "Blocked", "阻断项")}</div>
                      {scientistBlockedRequirements.slice(0, 5).map((item, index) => <div key={`${item}-${index}`}>- {item}</div>)}
                    </div>
                  ) : null}
                  {scientistOpenRequirements.length ? (
                    <div className="rounded border border-amber-100 bg-amber-50 p-2 text-amber-800">
                      <div className="mb-1 font-bold">{tx(locale, "Open", "待闭环")}</div>
                      {scientistOpenRequirements.slice(0, 6).map((item, index) => <div key={`${item}-${index}`}>- {item}</div>)}
                    </div>
                  ) : (
                    <div className="rounded border border-green-100 bg-green-50 p-2 text-green-800">
                      {tx(locale, "All tracked requirements are satisfied for this bounded turn.", "本次受控回合的已跟踪需求均已满足。")}
                    </div>
                  )}
                  {scientistNextEvidence.length ? (
                    <div>
                      <div className="mb-1 font-bold text-slate-500">{tx(locale, "Next Evidence", "下一步证据")}</div>
                      {scientistNextEvidence.slice(0, 6).map((item, index) => (
                        <div key={`${item}-${index}`} className="break-all rounded bg-slate-50 px-2 py-1 font-mono text-[11px] text-slate-700">
                          {item}
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="text-xs text-slate-400">{tx(locale, "Run an AI Scientist turn to derive the requirement ledger.", "运行一次 AI Scientist 回合以生成需求闭环清单。")}</div>
              )}
            </div>
          </div>
          <div className="lg:col-span-2 rounded-md border border-emerald-200 bg-emerald-50/50 p-3">
            <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <div className="text-xs font-bold uppercase text-emerald-800">
                  {tx(locale, "Evidence-Grounded Scientist Answer", "基于证据的科学家回答")}
                </div>
                <div className="mt-1 text-xs leading-5 text-slate-600">
                  {tx(
                    locale,
                    "Answers the requested research question after tool use, then records falsifiable hypotheses, comparison, selection rationale, and the next gated action.",
                    "工具调用完成后直接回答研究问题，并记录可证伪假设、对比、选择理由和下一步受控动作。"
                  )}
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <StatusBadge tone={(reasoningQuality?.score ?? 0) >= 85 ? "green" : (reasoningQuality?.score ?? 0) >= 70 ? "amber" : "red"}>
                  {reasoningQuality?.status ?? "not_run"} · {reasoningQuality?.score ?? 0}
                </StatusBadge>
                <StatusBadge tone={reasoningSynthesis?.llm?.used ? "blue" : "slate"}>
                  {reasoningSynthesis?.llm?.model ?? "deterministic fallback"}
                </StatusBadge>
              </div>
            </div>
            {reasoningSynthesis ? (
              <div className="space-y-3">
                <div className="rounded-md border border-white bg-white p-3 text-sm leading-6 text-slate-800">
                  {reasoningSynthesis.direct_answer || tx(locale, "No direct answer was produced.", "尚未生成直接回答。")}
                </div>
                <div className="grid gap-3 xl:grid-cols-[1.45fr_0.55fr]">
                  <div className="grid gap-2 md:grid-cols-3">
                    {reasoningHypotheses.slice(0, 6).map((hypothesis, index) => (
                      <div key={`${hypothesis.id ?? "hypothesis"}-${index}`} className="min-w-0 rounded-md border border-emerald-100 bg-white p-3">
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="font-mono text-[11px] font-bold text-emerald-700">{hypothesis.id ?? `H${index + 1}`}</div>
                            <div className="mt-1 text-sm font-bold text-slate-900">{hypothesis.title ?? "Untitled hypothesis"}</div>
                          </div>
                          <StatusBadge tone={hypothesis.risk === "high" ? "red" : hypothesis.risk === "medium" ? "amber" : "green"}>
                            {hypothesis.risk ?? "unknown"}
                          </StatusBadge>
                        </div>
                        <div className="mt-2 text-xs leading-5 text-slate-600">{hypothesis.mechanism ?? ""}</div>
                        <div className="mt-2 rounded border border-blue-100 bg-blue-50 p-2 text-xs leading-5 text-blue-900">
                          <span className="font-bold">{tx(locale, "Prediction", "可证伪预测")}:</span> {hypothesis.falsifiable_prediction ?? ""}
                        </div>
                        <div className="mt-2 rounded border border-red-100 bg-red-50 p-2 text-xs leading-5 text-red-800">
                          <span className="font-bold">{tx(locale, "Reject when", "否证条件")}:</span> {hypothesis.disconfirming_result ?? ""}
                        </div>
                        <div className="mt-2 text-[11px] text-slate-500">
                          evidence={hypothesis.evidence_strength ?? "unknown"} · cost={hypothesis.cost ?? "unknown"} · value={hypothesis.expected_value ?? "unknown"}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="rounded-md border border-emerald-100 bg-white p-3">
                    <div className="text-xs font-bold uppercase text-slate-500">{tx(locale, "Decision", "科研决策")}</div>
                    <div className="mt-2 font-mono text-sm font-bold text-emerald-800">
                      {reasoningSynthesis.selected_hypothesis_id ?? "(none)"}
                    </div>
                    <div className="mt-2 text-xs leading-5 text-slate-700">
                      {reasoningSynthesis.selected_rationale ?? ""}
                    </div>
                    <div className="mt-3 border-t border-slate-100 pt-3">
                      <div className="text-xs font-bold uppercase text-slate-500">{tx(locale, "Next Safe Action", "下一步安全动作")}</div>
                      <div className="mt-2 text-xs leading-5 text-slate-700">{reasoningNextAction?.action ?? ""}</div>
                      <div className="mt-2 break-all rounded bg-slate-950 px-2 py-2 font-mono text-xs text-white">
                        {reasoningNextAction?.command ?? "evomind briefing"}
                      </div>
                      <div className="mt-2 break-all font-mono text-[11px] text-amber-700">
                        gate={reasoningNextAction?.gate ?? "unknown"}
                      </div>
                    </div>
                    <div className="mt-3 border-t border-slate-100 pt-3 text-[11px] text-slate-500">
                      hypotheses={reasoningQuality?.hypotheses_produced ?? 0}/{reasoningQuality?.hypotheses_requested ?? 0}
                      {" · "}falsifiable={reasoningQuality?.complete_falsifiable_hypotheses ?? 0}
                      {" · "}cache={reasoningSynthesis.cache_hit ? "hit" : "miss"}
                      {" · "}ratio={Math.round((reasoningSynthesis.cache_stats?.hit_ratio ?? 0) * 100)}%
                    </div>
                    <div className="mt-2 break-all font-mono text-[10px] text-slate-500">
                      {reasoningSynthesis.artifact_path ?? ".xsci/scientist_reasoning_synthesis.json"}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="rounded border border-dashed border-emerald-200 bg-white/70 p-3 text-xs text-slate-500">
                {tx(locale, "Run a Scientist Turn to generate the evidence-grounded answer.", "运行一次科学家回合以生成基于证据的研究回答。")}
              </div>
            )}
          </div>
          <div className="lg:col-span-2 rounded-md border border-indigo-100 bg-indigo-50/60 p-3">
            <div className="mb-2 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-xs font-bold uppercase text-indigo-700">{tx(locale, "Scientific Critique", "科学家自评")}</div>
              <StatusBadge tone={(turnPlanCritique?.actionability_score ?? 0) >= 70 ? "green" : (turnPlanCritique?.actionability_score ?? 0) >= 45 ? "amber" : "red"}>
                {turnPlanCritique?.decision ?? "not_run"} · {turnPlanCritique?.actionability_score ?? 0}
              </StatusBadge>
            </div>
            <div className="grid gap-3 xl:grid-cols-3">
              <div className="rounded border border-white/80 bg-white/80 p-2">
                <div className="mb-1 text-[11px] font-bold uppercase text-slate-500">{tx(locale, "Evidence Gaps", "证据缺口")}</div>
                <div className="max-h-44 space-y-1.5 overflow-y-auto pr-1">
                  {turnPlanEvidenceGaps.length === 0 ? (
                    <div className="text-xs text-slate-400">{tx(locale, "No critique gap recorded yet.", "尚未记录自评缺口。")}</div>
                  ) : null}
                  {turnPlanEvidenceGaps.slice(0, 5).map((gap, index) => (
                    <div key={`${gap.gap ?? "gap"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs leading-4 text-slate-700">
                      <div className="flex items-start justify-between gap-2">
                        <span className="font-semibold text-slate-900">{gap.gap ?? "unknown_gap"}</span>
                        <StatusBadge tone={gap.severity === "blocking" || gap.severity === "high" ? "red" : gap.severity === "medium" ? "amber" : "slate"}>{gap.severity ?? "gap"}</StatusBadge>
                      </div>
                      <div className="mt-1">{gap.why_it_matters ?? ""}</div>
                      {gap.suggested_tool ? <div className="mt-1 font-mono text-[11px] text-indigo-700">tool={gap.suggested_tool}</div> : null}
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded border border-white/80 bg-white/80 p-2">
                <div className="mb-1 text-[11px] font-bold uppercase text-slate-500">{tx(locale, "Uncertainty", "不确定性")}</div>
                <div className="max-h-44 space-y-1.5 overflow-y-auto pr-1">
                  {turnPlanUncertainty.length === 0 ? (
                    <div className="text-xs text-slate-400">{tx(locale, "No uncertainty driver recorded yet.", "尚未记录不确定性来源。")}</div>
                  ) : null}
                  {turnPlanUncertainty.slice(0, 5).map((item, index) => (
                    <div key={`${item}-${index}`} className="rounded border border-amber-100 bg-amber-50 px-2 py-1.5 text-xs leading-4 text-amber-800">
                      {item}
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded border border-white/80 bg-white/80 p-2">
                <div className="mb-1 text-[11px] font-bold uppercase text-slate-500">{tx(locale, "Claim Boundaries", "声明边界")}</div>
                <div className="max-h-44 space-y-1.5 overflow-y-auto pr-1">
                  {turnPlanClaimBoundaries.length === 0 ? (
                    <div className="text-xs text-slate-400">{tx(locale, "No claim boundary recorded yet.", "尚未记录声明边界。")}</div>
                  ) : null}
                  {turnPlanClaimBoundaries.slice(0, 4).map((item, index) => (
                    <div key={`${item}-${index}`} className="rounded border border-red-100 bg-red-50 px-2 py-1.5 text-xs leading-4 text-red-800">
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <div className="lg:col-span-2 rounded-md border border-slate-200 bg-white p-3">
            <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="text-xs font-bold uppercase text-slate-500">{tx(locale, "Scientist Lifecycle", "科学家生命周期")}</div>
                <div className="mt-1 text-xs leading-5 text-slate-600">
                  {tx(
                    locale,
                    "A Codex/Claude-style turn must leave evidence for observe, plan, act, reflect, and improve.",
                    "类 Codex / Claude Code 的研究回合必须留下观察、计划、行动、反思、改进五段证据。"
                  )}
                </div>
              </div>
              <StatusBadge tone={scientistParityGateTone}>
                {scientistParityPassedCount}/{scientistParityPhases.length || 5}
              </StatusBadge>
            </div>
            <div className="grid gap-2 md:grid-cols-5">
              {scientistParityPhases.slice(0, 5).map((phase, index) => {
                const executed = Array.isArray(phase.executed_tools) ? phase.executed_tools : [];
                const deferred = Array.isArray(phase.deferred_tools) ? phase.deferred_tools : [];
                const gaps = Array.isArray(phase.evidence_gaps) ? phase.evidence_gaps : [];
                return (
                  <div key={`${phase.phase ?? "phase"}-${index}`} className="min-w-0 rounded-md border border-slate-100 bg-slate-50 px-2 py-2">
                    <div className="flex items-start justify-between gap-2">
                      <span className="break-all font-mono text-xs font-bold text-slate-900">{phase.phase ?? `phase-${index + 1}`}</span>
                      <StatusBadge tone={parityPhaseTone(phase.status)}>{phase.status ?? "not_run"}</StatusBadge>
                    </div>
                    <div className="mt-2 line-clamp-4 text-[11px] leading-4 text-slate-600">{phase.purpose ?? ""}</div>
                    {phase.gate ? <div className="mt-2 break-all font-mono text-[10px] text-blue-700">gate={phase.gate}</div> : null}
                    {phase.next_safe_command ? <div className="mt-2 break-all font-mono text-[10px] text-indigo-700">{phase.next_safe_command}</div> : null}
                    {executed.length ? <div className="mt-2 text-[10px] font-semibold text-emerald-700">{tx(locale, "tools", "工具")}={executed.slice(0, 3).join(", ")}</div> : null}
                    {deferred.length ? <div className="mt-2 text-[10px] font-semibold text-amber-700">{tx(locale, "deferred", "延期")}={deferred.slice(0, 3).join(", ")}</div> : null}
                    {gaps.length ? <div className="mt-2 text-[10px] font-semibold text-red-700">{tx(locale, "gaps", "缺口")}={gaps.length}</div> : null}
                  </div>
                );
              })}
            </div>
            <div className="mt-3 grid gap-2 text-[11px] sm:grid-cols-3">
              <div className="break-all rounded border border-slate-100 bg-slate-50 px-2 py-1.5 font-mono text-slate-700">
                schema={scientistParityLifecycle?.schema ?? "not_recorded"}
              </div>
              <div className="break-all rounded border border-slate-100 bg-slate-50 px-2 py-1.5 font-mono text-slate-700">
                artifact={scientistParityArtifact}
              </div>
              <div className="break-all rounded border border-slate-100 bg-slate-50 px-2 py-1.5 font-mono text-slate-700">
                training={scientistParityLifecycle?.no_training_started === false ? "started" : "not_started"} · submit={scientistParityLifecycle?.official_submit ?? "blocked"}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card data-testid="verified-upgrade-campaign">
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Verified Upgrade Campaign", "可信升级活动")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "External hidden-suite certification and the active champion campaign are evaluated as separate release gates.",
                "外部隐藏测试认证与当前冠军升级活动分别作为独立发布门禁。"
              )}
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => void runScientistUpgradeCampaignAction("status")}
              disabled={scientistUpgradeCampaignBusy}
              title={tx(locale, "Refresh campaign status", "刷新升级状态")}
            >
              <RefreshCcw className={`h-4 w-4 ${scientistUpgradeCampaignBusy ? "animate-spin" : ""}`} />
              {tx(locale, "Refresh", "刷新")}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => void runScientistUpgradeCampaignAction("run")}
              disabled={scientistUpgradeCampaignBusy}
            >
              <Play className="h-4 w-4" />
              {tx(locale, "Evaluate Candidates", "评估候选")}
            </Button>
            <Button
              size="sm"
              onClick={() => void runScientistUpgradeCampaignAction("promote")}
              disabled={scientistUpgradeCampaignBusy || !scientistPromotionApproved || scientistCampaignStatus !== "awaiting_human_promotion"}
            >
              <ShieldCheck className="h-4 w-4" />
              {tx(locale, "Promote", "晋升")}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => void runScientistUpgradeCampaignAction("rollback")}
              disabled={scientistUpgradeCampaignBusy || scientistCampaignStatus !== "active"}
            >
              <History className="h-4 w-4" />
              {tx(locale, "Rollback", "回滚")}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.78fr_1.22fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Parity Gate", "对标门禁")} value={<StatusBadge tone={scientistCampaignTone}>{scientistParityCertified ? "certified" : "blocked"}</StatusBadge>} />
            <Row label={tx(locale, "Certification", "外部认证")} value={scientistCertificationStatus} />
            <Row label={tx(locale, "Campaign", "升级活动")} value={scientistCampaignStatus} />
            <Row label={tx(locale, "Score Cap", "评分上限")} value={scientistUpgradeCampaign?.score_cap ?? 84} />
            <Row label={tx(locale, "Champion Ref", "冠军引用")} value={String(scientistCampaignEvidence.champion_ref ?? "refs/evomind/champion")} />
            <Row label={tx(locale, "Canary", "运行时探针")} value={String(scientistCampaignEvidence.promotion_verified === true)} />
            <Row label={tx(locale, "Rollback Proof", "回滚证明")} value={String(scientistCampaignEvidence.rollback_verified === true)} />
            <label className="mt-3 flex items-start gap-2 border-t border-slate-200 pt-3 text-xs text-slate-700">
              <input
                type="checkbox"
                className="mt-0.5 h-4 w-4"
                checked={scientistPromotionApproved}
                onChange={(event) => setScientistPromotionApproved(event.target.checked)}
              />
              <span>{tx(locale, "I approve promotion of the selected strictly improved candidate.", "我确认晋升已通过严格提升门禁的候选版本。")}</span>
            </label>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Blocking Evidence", "阻断证据")}</div>
              <div className="space-y-2">
                {(scientistUpgradeCampaign?.blockers ?? []).length === 0 ? (
                  <div className="text-xs text-emerald-700">{tx(locale, "No parity blockers are active.", "当前没有对标阻断项。")}</div>
                ) : null}
                {(scientistUpgradeCampaign?.blockers ?? []).map((blocker) => (
                  <div key={blocker} className="break-all rounded border border-rose-100 bg-rose-50 px-2 py-1.5 font-mono text-[11px] text-rose-800">{blocker}</div>
                ))}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Bound Evidence", "绑定证据")}</div>
              <Row label={tx(locale, "Result Digest", "结果摘要")} value={String(scientistCertificationEvidence.result_sha256 ?? "(none)")} />
              <Row label={tx(locale, "Exact Source", "精确源码")} value={String(scientistCertificationEvidence.source_identity_matches === true)} />
              <Row label={tx(locale, "Strict Improvement", "严格提升")} value={String(scientistCampaignEvidence.strict_improvement_verified === true)} />
              <Row label={tx(locale, "Champion Match", "冠军匹配")} value={String(scientistCampaignEvidence.champion_ref_matches === true)} />
            </div>
            <div className="max-h-80 overflow-auto rounded-md md:col-span-2">
              <JsonInspector data={{ certification: scientistCertificationEvidence, campaign: scientistCampaignEvidence }} />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Context Packet", "科学家上下文包")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Compacts task, readiness gates, memory, active strategy, requirement ledger, artifacts, and the next safe command before the agent answers.",
                "在智能体回答前压缩任务、就绪门禁、记忆、当前策略、需求清单、证据文件和下一条安全命令。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistContextPacket()} disabled={scientistContextPacketBusy}>
            {scientistContextPacketBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <BrainCircuit className="h-4 w-4" />}
            {tx(locale, "Build Context", "生成上下文")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.72fr_1.28fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Quality", "上下文质量")}
              value={<StatusBadge tone={(scientistContextPacket?.context_quality?.score ?? 0) >= 75 ? "green" : (scientistContextPacket?.context_quality?.score ?? 0) >= 55 ? "amber" : "slate"}>{scientistContextPacket?.context_quality?.score ?? 0}</StatusBadge>}
            />
            <Row label={tx(locale, "Mode", "状态")} value={scientistContextPacket?.context_quality?.interpretation ?? "not_run"} />
            <Row label={tx(locale, "Task", "任务")} value={scientistContextPacket?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Can Execute", "可执行训练")} value={scientistContextPacket?.readiness?.can_execute ? "true" : "false"} />
            <Row label={tx(locale, "Compute", "算力")} value={scientistContextPacket?.readiness?.compute_backend ?? "unknown"} />
            <Row label={tx(locale, "Strategy", "当前策略")} value={scientistContextPacket?.active_strategy?.selected_action || "none"} />
            <Row label={tx(locale, "Strategy Gate", "策略门禁")} value={scientistContextPacket?.active_strategy?.gate_status || "unknown"} />
            <Row label={tx(locale, "Next", "下一步")} value={scientistContextPacket?.next_safe_command ?? "evomind briefing"} />
            <Row label={tx(locale, "Memory", "记忆")} value={scientistContextPacket?.memory_digest?.retrospective_records ?? 0} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistContextPacket?.artifact_path ?? ".xsci/scientist_context_packet.json"} />
            <Row label={tx(locale, "Markdown", "Markdown")} value={scientistContextPacket?.markdown_artifact_path ?? ".xsci/scientist_context_packet.md"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistContextPacket?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistContextPacket?.official_submit ?? "blocked_until_explicit_human_approval"} />
          </div>
          <div className="grid gap-3 md:grid-cols-[1fr_1fr]">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Blocking Gates", "阻塞门禁")}</div>
              <div className="max-h-36 space-y-2 overflow-y-auto pr-1">
                {(scientistContextPacket?.readiness?.blocking_gates ?? []).length === 0 ? (
                  <div className="text-xs text-emerald-700">{tx(locale, "No blocking gate in the latest context packet.", "最新上下文包没有硬阻塞门禁。")}</div>
                ) : (
                  (scientistContextPacket?.readiness?.blocking_gates ?? []).slice(0, 6).map((gate, index) => (
                    <div key={`${gate}-${index}`} className="rounded border border-amber-100 bg-amber-50 px-2 py-1.5 text-xs text-amber-900">
                      {gate}
                    </div>
                  ))
                )}
              </div>
              <div className="mt-3 border-t border-slate-100 pt-2">
                <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Recent Lessons", "近期经验")}</div>
                <div className="max-h-40 space-y-2 overflow-y-auto pr-1">
                  {(scientistContextPacket?.memory_digest?.recent_lessons ?? []).length === 0 ? (
                    <div className="text-xs text-slate-400">{tx(locale, "No reusable lesson loaded yet.", "尚未加载可复用经验。")}</div>
                  ) : (
                    (scientistContextPacket?.memory_digest?.recent_lessons ?? []).slice(0, 5).map((lesson, index) => (
                      <div key={`${lesson}-${index}`} className="rounded border border-blue-100 bg-blue-50 px-2 py-1.5 text-xs leading-4 text-blue-900">
                        {lesson}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Requirement Context", "需求上下文")}</div>
              <JsonInspector
                data={{
                  open_requirements: (scientistContextPacket?.requirement_context?.open_requirements as unknown[] | undefined)?.slice?.(0, 8) ?? [],
                  blocked_requirements: (scientistContextPacket?.requirement_context?.blocked_requirements as unknown[] | undefined)?.slice?.(0, 8) ?? [],
                  execution_partition: scientistContextPacket?.requirement_context?.execution_partition ?? {}
                }}
              />
              <div className="mt-3 border-t border-slate-100 pt-2">
                <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Artifact Inventory", "证据清单")}</div>
                <div className="grid gap-1 text-[11px]">
                  {(scientistContextPacket?.artifact_inventory ?? []).slice(0, 8).map((artifact) => (
                    <div key={artifact.name ?? artifact.path} className="flex items-center justify-between gap-2 rounded bg-slate-50 px-2 py-1">
                      <span className="truncate font-mono text-slate-700">{artifact.name ?? artifact.path}</span>
                      <StatusBadge tone={artifact.present ? "green" : "slate"}>{artifact.present ? "present" : "missing"}</StatusBadge>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Strategy Optimizer", "科学家策略优化")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Ranks safe interventions by expected impact, evidence strength, cost, risk, and gate status before choosing the next command.",
                "把安全干预动作按预期影响、证据强度、成本、风险和门禁状态排序，再选择下一条命令。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistStrategyOptimizer()} disabled={scientistStrategyOptimizerBusy}>
            {scientistStrategyOptimizerBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <GitBranch className="h-4 w-4" />}
            {tx(locale, "Optimize Strategy", "优化策略")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.72fr_1.28fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Posture", "姿态")} value={<StatusBadge tone={readinessTone(scientistStrategyOptimizer?.strategy_posture)}>{scientistStrategyOptimizer?.strategy_posture ?? "not_run"}</StatusBadge>} />
            <Row label={tx(locale, "Source", "来源姿态")} value={scientistStrategyOptimizer?.source_posture ?? "unknown"} />
            <Row label={tx(locale, "Candidates", "候选数")} value={scientistStrategyOptimizer?.intervention_ranking?.length ?? 0} />
            <Row label={tx(locale, "Selected", "选中策略")} value={scientistStrategyOptimizer?.selected_strategy?.id ?? "none"} />
            <Row label={tx(locale, "Score", "分数")} value={scientistStrategyOptimizer?.selected_strategy?.total_score ?? 0} />
            <Row label={tx(locale, "Gate", "门禁")} value={scientistStrategyOptimizer?.selected_strategy?.gate_status ?? "not_run"} />
            <Row label={tx(locale, "Next", "下一步")} value={scientistStrategyOptimizer?.next_safe_command ?? "evomind strategy"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistStrategyOptimizer?.artifact_path ?? ".xsci/scientist_strategy_optimizer.json"} />
            <Row label={tx(locale, "Markdown", "Markdown")} value={scientistStrategyOptimizer?.markdown_artifact_path ?? ".xsci/scientist_strategy_optimizer.md"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistStrategyOptimizer?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistStrategyOptimizer?.official_submit ?? "blocked_until_explicit_human_approval"} />
          </div>
          <div className="grid gap-3 md:grid-cols-[0.9fr_1.1fr]">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Selected Strategy", "选中策略")}</div>
              {scientistStrategyOptimizer?.selected_strategy ? (
                <div className="space-y-2 rounded border border-blue-100 bg-blue-50 px-3 py-2 text-xs">
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-bold text-blue-950">{scientistStrategyOptimizer.selected_strategy.title ?? scientistStrategyOptimizer.selected_strategy.id}</div>
                    <StatusBadge tone={scientistStrategyOptimizer.selected_strategy.gate_status === "safe_read_only" ? "green" : "amber"}>
                      {scientistStrategyOptimizer.selected_strategy.total_score ?? 0}
                    </StatusBadge>
                  </div>
                  <div className="font-mono text-[11px] text-blue-800">{scientistStrategyOptimizer.selected_strategy.safe_next_command ?? ""}</div>
                  <div className="leading-4 text-blue-800">{scientistStrategyOptimizer.selected_strategy.rationale ?? ""}</div>
                  <div className="flex flex-wrap gap-1">
                    <span className="rounded bg-white px-1.5 py-0.5 text-[11px] text-blue-800">impact={scientistStrategyOptimizer.selected_strategy.expected_impact ?? 0}</span>
                    <span className="rounded bg-white px-1.5 py-0.5 text-[11px] text-blue-800">evidence={scientistStrategyOptimizer.selected_strategy.evidence_strength ?? 0}</span>
                    <span className="rounded bg-white px-1.5 py-0.5 text-[11px] text-blue-800">risk={scientistStrategyOptimizer.selected_strategy.risk_level ?? "unknown"}</span>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-slate-400">{tx(locale, "Run strategy optimizer to select a next action.", "运行策略优化后显示选中策略。")}</div>
              )}
              <div className="mt-3 border-t border-slate-100 pt-2">
                <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Decision Matrix", "决策矩阵")}</div>
                <JsonInspector data={scientistStrategyOptimizer?.decision_matrix ?? { candidate_count: 0 }} />
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Ranked Interventions", "干预排序")}</div>
              <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                {(scientistStrategyOptimizer?.intervention_ranking ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No ranked interventions yet.", "暂无排序干预。")}</div>
                ) : (
                  scientistStrategyOptimizer?.intervention_ranking?.slice(0, 8).map((item, index) => (
                    <div key={`${item.id ?? "strategy"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <span className="font-bold text-slate-800">#{item.rank ?? index + 1} {item.title ?? item.id}</span>
                        <StatusBadge tone={item.gate_status === "safe_read_only" ? "green" : item.gate_status?.includes("blocked") ? "amber" : "blue"}>{item.total_score ?? 0}</StatusBadge>
                      </div>
                      <div className="mt-1 font-mono text-[11px] text-slate-700">{item.safe_next_command ?? ""}</div>
                      <div className="mt-1 grid grid-cols-3 gap-1 text-[11px] text-slate-500">
                        <span>impact={item.expected_impact ?? 0}</span>
                        <span>evidence={item.evidence_strength ?? 0}</span>
                        <span>risk={item.risk_level ?? "unknown"}</span>
                      </div>
                      <div className="mt-1 leading-4 text-slate-500">{item.rationale ?? ""}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Failure-to-Patch Work Order", "失败转补丁工单")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Turns the latest Scientist failure or blocker evidence into a code-agent work order. If the blocker is external resources or data, source edits stay blocked.",
                "把最近一次科学家失败或阻塞证据转成代码 Agent 修复工单；如果问题是外部资源或数据门禁，会阻止无意义源码修改。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistPatchWorkOrder()} disabled={scientistPatchWorkOrderBusy}>
            {scientistPatchWorkOrderBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
            {tx(locale, "Create Patch Order", "生成补丁工单")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.72fr_1.28fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Status", "状态")} value={<StatusBadge tone={scientistPatchTone}>{scientistPatchStatus}</StatusBadge>} />
            <Row label={tx(locale, "Issue", "问题")} value={scientistPatchWorkOrder?.selected_issue_id || scientistPatchOrderBody?.issue_id || "(none)"} />
            <Row label={tx(locale, "Title", "标题")} value={scientistPatchWorkOrder?.selected_title || scientistPatchOrderBody?.title || "(none)"} />
            <Row label={tx(locale, "Task", "任务")} value={scientistPatchWorkOrder?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Safe Next", "安全下一步")} value={scientistPatchOrderBody?.safe_next_command ?? scientistPatchWorkOrder?.next_safe_commands?.[0] ?? "evomind patch-order"} />
            <Row label={tx(locale, "Human Gate", "人工门禁")} value={scientistPatchOrderBody?.human_gate ?? "blocked_until_code_agent_or_human_review_applies_patch"} />
            <Row label={tx(locale, "Work Order", "工单文件")} value={scientistPatchWorkOrder?.artifact_path ?? ".xsci/scientist_patch_work_order.json"} />
            <Row label={tx(locale, "Queue", "行动队列")} value={scientistPatchWorkOrder?.action_queue_path ?? ".xsci/scientist_patch_action_queue.json"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistPatchWorkOrder?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistPatchWorkOrder?.official_submit ?? "blocked_until_explicit_human_approval"} />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Patch Scope", "修复范围")}</div>
              <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                {(scientistPatchOrderBody?.files_to_edit ?? []).length === 0 ? (
                  <div className="rounded border border-amber-100 bg-amber-50 px-2 py-1.5 text-xs leading-4 text-amber-800">
                    {scientistPatchStatus === "blocked_external_gate"
                      ? tx(locale, "No source patch is allowed because the latest evidence points to an external gate.", "最近证据指向外部资源门禁，当前不允许生成源码补丁。")
                      : tx(locale, "No editable file scope has been selected yet.", "尚未选中可编辑文件范围。")}
                  </div>
                ) : null}
                {(scientistPatchOrderBody?.files_to_edit ?? []).slice(0, 8).map((file) => (
                  <div key={file} className="break-all rounded border border-blue-100 bg-blue-50 px-2 py-1.5 font-mono text-[11px] text-blue-800">{file}</div>
                ))}
                {(scientistPatchOrderBody?.files_to_inspect ?? []).slice(0, 8).map((file) => (
                  <div key={file} className="break-all rounded border border-slate-100 bg-slate-50 px-2 py-1.5 font-mono text-[11px] text-slate-700">{file}</div>
                ))}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Acceptance & Code Agent Prompt", "验收与代码 Agent 提示")}</div>
              <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                {(scientistPatchOrderBody?.acceptance_checks ?? []).slice(0, 6).map((check) => (
                  <div key={check} className="break-all rounded border border-slate-100 bg-slate-50 px-2 py-1.5 font-mono text-[11px] text-slate-700">{check}</div>
                ))}
                {scientistPatchOrderBody?.code_agent_prompt ? (
                  <div className="rounded border border-indigo-100 bg-indigo-50 px-2 py-1.5 text-xs leading-4 text-indigo-900">
                    {scientistPatchOrderBody.code_agent_prompt}
                  </div>
                ) : null}
                {scientistPatchOrderBody?.rationale ? (
                  <div className="rounded border border-slate-100 bg-white px-2 py-1.5 text-xs leading-4 text-slate-600">
                    {scientistPatchOrderBody.rationale}
                  </div>
                ) : null}
                {(scientistPatchWorkOrder?.next_safe_commands ?? []).slice(0, 4).map((command) => (
                  <div key={command} className="break-all rounded border border-blue-100 bg-blue-50 px-2 py-1.5 font-mono text-[11px] font-semibold text-blue-800">{command}</div>
                ))}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3 md:col-span-2">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-xs font-bold uppercase text-slate-500">{tx(locale, "Action Queue & Evidence", "行动队列与证据")}</div>
                <StatusBadge tone={scientistPatchActions.length ? "blue" : "slate"}>{scientistPatchActions.length}</StatusBadge>
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                <div className="max-h-56 space-y-2 overflow-y-auto pr-1">
                  {scientistPatchActions.length === 0 ? <div className="text-xs text-slate-400">{tx(locale, "No patch action queue yet.", "暂无补丁行动队列。")}</div> : null}
                  {scientistPatchActions.slice(0, 6).map((action, index) => (
                    <div key={`${action.id ?? "patch-action"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <span className="font-bold text-slate-900">{action.title ?? action.id ?? "patch action"}</span>
                        <StatusBadge tone={action.status === "ready_for_code_agent" ? "green" : action.status === "blocked_external_gate" ? "red" : "slate"}>{action.status ?? "queued"}</StatusBadge>
                      </div>
                      {action.command ? <div className="mt-1 break-all font-mono text-[11px] text-blue-700">{action.command}</div> : null}
                      {action.gate ? <div className="mt-1 text-[11px] text-slate-500">gate={action.gate}</div> : null}
                    </div>
                  ))}
                </div>
                <div className="max-h-56 space-y-1 overflow-y-auto pr-1">
                  {(scientistPatchOrderBody?.expected_artifacts ?? scientistPatchWorkOrder?.source_artifacts ?? []).slice(0, 10).map((artifact) => (
                    <div key={artifact} className="break-all rounded border border-slate-100 bg-white px-2 py-1 font-mono text-[11px] text-slate-700">{artifact}</div>
                  ))}
                  {scientistPatchWorkOrder?.trials_path ? (
                    <div className="break-all rounded border border-emerald-100 bg-emerald-50 px-2 py-1 font-mono text-[11px] text-emerald-800">{scientistPatchWorkOrder.trials_path}</div>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Isolated Engineering Loop", "隔离工程执行闭环")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Generates or validates a Code Agent diff inside a detached Git worktree, runs allowlisted acceptance checks, proves the main worktree stayed unchanged, and stops before merge.",
                "在独立 Git worktree 中生成或验证 Code Agent 补丁，运行白名单验收测试，证明主工作区未被修改，并在合并前停止。"
              )}
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" variant="secondary" onClick={() => void runScientistEngineeringLoop(false)} disabled={scientistEngineeringBusy}>
              {scientistEngineeringBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
              {tx(locale, "Validate Patch", "验证已有补丁")}
            </Button>
            <Button size="sm" onClick={() => void runScientistEngineeringLoop(true)} disabled={scientistEngineeringBusy}>
              {scientistEngineeringBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <BrainCircuit className="h-4 w-4" />}
              {tx(locale, "Generate + Validate", "生成并隔离验证")}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.72fr_1.28fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Status", "状态")} value={<StatusBadge tone={scientistEngineeringTone}>{scientistEngineeringStatus}</StatusBadge>} />
            <Row label={tx(locale, "Task", "任务")} value={scientistEngineeringLoop?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Checks", "验收测试")} value={`${scientistEngineeringPassed}/${scientistEngineeringChecks.length}`} />
            <Row label={tx(locale, "Patch Applied", "隔离应用补丁")} value={String(scientistEngineeringLoop?.patch_applied_in_isolated_worktree ?? false)} />
            <Row label={tx(locale, "Main Changed", "主工作区被修改")} value={String(scientistEngineeringLoop?.main_worktree_modified ?? false)} />
            <Row label={tx(locale, "Merge Ready", "可进入合并审查")} value={String(scientistEngineeringLoop?.merge_ready ?? false)} />
            <Row label={tx(locale, "Human Gate", "人工门禁")} value={scientistEngineeringLoop?.work_order?.human_gate ?? scientistEngineeringLoop?.human_gate ?? "review_candidate_before_merge"} />
            <Row label={tx(locale, "Next", "下一步")} value={scientistEngineeringLoop?.next_safe_command ?? "evomind patch-order"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistEngineeringLoop?.artifact_path ?? ".xsci/scientist_engineering_loop.json"} />
            <Row label={tx(locale, "Candidate Diff", "候选补丁")} value={scientistEngineeringLoop?.candidate_diff_path ?? "(none)"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistEngineeringLoop?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistEngineeringLoop?.official_submit ?? "blocked_until_explicit_human_approval"} />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Patch Scope", "补丁范围")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {(scientistEngineeringLoop?.changed_files ?? []).length === 0 ? (
                  <div className="text-xs leading-5 text-slate-500">
                    {scientistEngineeringLoop?.message ?? tx(locale, "No isolated engineering run yet.", "尚未运行隔离工程验证。")}
                  </div>
                ) : null}
                {(scientistEngineeringLoop?.changed_files ?? []).slice(0, 12).map((file) => (
                  <div key={file} className="break-all rounded border border-blue-100 bg-blue-50 px-2 py-1.5 font-mono text-[11px] text-blue-800">{file}</div>
                ))}
                {scientistEngineeringLoop?.patch_path ? (
                  <div className="break-all rounded border border-slate-100 bg-slate-50 px-2 py-1.5 font-mono text-[11px] text-slate-700">{scientistEngineeringLoop.patch_path}</div>
                ) : null}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Acceptance Checks", "隔离验收测试")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {scientistEngineeringChecks.length === 0 ? <div className="text-xs text-slate-400">{tx(locale, "No checks recorded.", "尚未记录验收测试。")}</div> : null}
                {scientistEngineeringChecks.slice(0, 10).map((check, index) => (
                  <div key={`${check.command ?? "check"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs">
                    <div className="flex items-start justify-between gap-2">
                      <span className="break-all font-mono text-[11px] text-slate-800">{check.command ?? "unknown check"}</span>
                      <StatusBadge tone={check.passed ? "green" : "red"}>{check.passed ? "pass" : "fail"}</StatusBadge>
                    </div>
                    {check.log_path ? <div className="mt-1 break-all font-mono text-[10px] text-slate-500">{check.log_path}</div> : null}
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-md border border-emerald-100 bg-emerald-50 p-3 md:col-span-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="text-xs font-bold uppercase text-emerald-800">{tx(locale, "Isolation Evidence", "隔离与回滚证据")}</div>
                  <div className="mt-1 text-xs leading-5 text-emerald-900">
                    {scientistEngineeringLoop?.main_worktree_modified
                      ? tx(locale, "Main worktree protection failed. Candidate must not be merged.", "主工作区保护失败，候选补丁不得合并。")
                      : tx(locale, "The main worktree remains unchanged. A passing candidate still requires human review before merge.", "主工作区保持不变；即使候选通过测试，仍需人工审查后才能合并。")}
                  </div>
                </div>
                <StatusBadge tone={scientistEngineeringLoop?.merge_ready ? "green" : "amber"}>
                  {scientistEngineeringLoop?.epistemic_status ?? "not_validated"}
                </StatusBadge>
              </div>
              {scientistEngineeringLoop?.run_manifest_path ? (
                <div className="mt-2 break-all font-mono text-[11px] text-emerald-800">{scientistEngineeringLoop.run_manifest_path}</div>
              ) : null}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Self-Upgrade Work Order", "科学家自升级工单")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Selects the next P0 capability gap and creates a code-agent work order, action queue, trace, and lesson. It never edits code, trains, downloads, or submits.",
                "选择下一个 P0 能力缺口，并生成代码 Agent 工单、行动队列、轨迹和经验记录。它不会修改代码、训练、下载或提交。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistSelfUpgradeLoop()} disabled={scientistSelfUpgradeBusy}>
            {scientistSelfUpgradeBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
            {tx(locale, "Create Work Order", "生成自升级工单")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.72fr_1.28fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Status", "状态")}
              value={<StatusBadge tone={scientistSelfUpgradeLoop?.status === "ready_for_code_agent" ? "green" : scientistSelfUpgradeLoop?.status === "no_open_upgrade_backlog" ? "blue" : "slate"}>{scientistSelfUpgradeLoop?.status ?? "not_run"}</StatusBadge>}
            />
            <Row label={tx(locale, "Selected Backlog", "选中缺口")} value={scientistSelfUpgradeLoop?.selected_backlog_id || "(none)"} />
            <Row label={tx(locale, "Title", "标题")} value={scientistSelfUpgradeLoop?.selected_title || "(none)"} />
            <Row label={tx(locale, "Open Backlog", "待处理缺口")} value={scientistSelfUpgradeLoop?.open_backlog_count ?? 0} />
            <Row label={tx(locale, "Audit Score Before", "升级前审计分")} value={scientistSelfUpgradeLoop?.overall_score_before ?? scientistSelfAudit?.overall_score ?? 0} />
            <Row label={tx(locale, "Work Order", "工单文件")} value={scientistSelfUpgradeLoop?.work_order_path ?? ".xsci/scientist_self_upgrade_work_order.json"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistSelfUpgradeLoop?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistSelfUpgradeLoop?.official_submit ?? "blocked_until_explicit_human_approval"} />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Files & Acceptance", "文件与验收")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {(scientistSelfUpgradeLoop?.work_order?.files_to_edit ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No work order yet.", "尚未生成自升级工单。")}</div>
                ) : null}
                {(scientistSelfUpgradeLoop?.work_order?.files_to_edit ?? []).slice(0, 8).map((path) => (
                  <div key={path} className="break-all rounded border border-blue-100 bg-blue-50 px-2 py-1.5 font-mono text-[11px] text-blue-800">{path}</div>
                ))}
                {(scientistSelfUpgradeLoop?.work_order?.acceptance_checks ?? []).slice(0, 6).map((check) => (
                  <div key={check} className="break-all rounded border border-slate-100 bg-slate-50 px-2 py-1.5 font-mono text-[11px] text-slate-700">{check}</div>
                ))}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Evidence Chain", "证据链")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {(scientistSelfUpgradeLoop?.loop_phases ?? []).slice(0, 6).map((phase, index) => (
                  <div key={`${String(phase.phase ?? "phase")}-${index}`} className="rounded border border-emerald-100 bg-emerald-50 px-2 py-1.5 text-xs">
                    <div className="font-bold text-emerald-900">{String(phase.phase ?? `phase-${index + 1}`)}</div>
                    <div className="break-all font-mono text-[11px] text-emerald-800">{String(phase.artifact ?? "(none)")}</div>
                  </div>
                ))}
                {(scientistSelfUpgradeLoop?.next_safe_commands ?? []).slice(0, 4).map((command) => (
                  <div key={command} className="break-all rounded border border-slate-100 bg-white px-2 py-1.5 font-mono text-[11px] text-slate-700">{command}</div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Upgrade Plan", "科学家升级计划")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Turns the self-audit backlog into a concrete engineering plan: files, gates, artifacts, acceptance checks, and safe commands.",
                "把自我审计 backlog 转成具体工程计划：文件、门禁、证据、验收检查和安全命令。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistUpgradePlan()} disabled={scientistUpgradePlanBusy}>
            {scientistUpgradePlanBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <GitBranch className="h-4 w-4" />}
            {tx(locale, "Build Upgrade Plan", "生成升级计划")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.78fr_1.22fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Readiness", "状态")}
              value={<StatusBadge tone={scientistUpgradePlan?.readiness === "ready_for_engineering_review" ? "green" : scientistUpgradePlan?.readiness === "no_open_upgrade_backlog" ? "blue" : "slate"}>{scientistUpgradePlan?.readiness ?? "not_run"}</StatusBadge>}
            />
            <Row label={tx(locale, "Open Backlog", "待处理升级项")} value={scientistUpgradePlan?.open_backlog_count ?? 0} />
            <Row label={tx(locale, "Self-Audit Score", "自审计评分")} value={scientistUpgradePlan?.self_audit_score ?? scientistSelfAudit?.overall_score ?? 0} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistUpgradePlan?.artifact_path ?? ".xsci/scientist_upgrade_plan.json"} />
            <Row label={tx(locale, "Source Backlog", "来源 Backlog")} value={scientistUpgradePlan?.source_backlog_path ?? ".xsci/scientist_upgrade_backlog.json"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistUpgradePlan?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistUpgradePlan?.official_submit ?? "blocked_until_explicit_human_approval"} />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Planned Engineering Steps", "工程步骤")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {(scientistUpgradePlan?.planned_steps ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "Run Upgrade Plan to convert backlog into implementation steps.", "运行升级计划后会把 backlog 转成实施步骤。")}</div>
                ) : (
                  scientistUpgradePlan?.planned_steps?.slice(0, 6).map((step, index) => (
                    <div key={`${step.step_id ?? step.id ?? "step"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <span className="font-bold text-slate-800">{step.title ?? step.backlog_id ?? step.id ?? "upgrade step"}</span>
                        <StatusBadge tone={step.priority === "P0" ? "red" : step.priority === "P1" ? "amber" : "slate"}>{step.priority ?? "P?"}</StatusBadge>
                      </div>
                      {(step.files_to_edit ?? step.files_to_inspect)?.length ? <div className="mt-1 break-all font-mono text-[11px] text-blue-700">{(step.files_to_edit ?? step.files_to_inspect ?? []).slice(0, 3).join(" | ")}</div> : null}
                      {step.safe_next_command ? <div className="mt-1 font-mono text-[11px] text-slate-600">{step.safe_next_command}</div> : null}
                    </div>
                  ))
                )}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Acceptance & Next Commands", "验收与下一步")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {(scientistUpgradePlan?.planned_steps ?? []).slice(0, 4).map((step, index) => (
                  <div key={`${step.step_id ?? step.id ?? "accept"}-${index}`} className="rounded border border-slate-100 bg-white px-2 py-1.5 text-xs">
                    <div className="font-bold text-slate-800">{step.step_id ?? step.backlog_id ?? step.id ?? `step-${index + 1}`}</div>
                    {(step.acceptance_checks ?? []).slice(0, 4).map((check) => (
                      <div key={check} className="mt-1 break-all font-mono text-[11px] text-slate-600">{check}</div>
                    ))}
                    {(step.expected_artifacts ?? []).slice(0, 3).map((artifact) => (
                      <div key={artifact} className="mt-1 break-all font-mono text-[11px] text-emerald-700">{artifact}</div>
                    ))}
                  </div>
                ))}
                {(scientistUpgradePlan?.next_safe_commands ?? []).length ? (
                  <div className="rounded border border-blue-100 bg-blue-50 px-2 py-1.5 text-xs">
                    <div className="mb-1 font-bold uppercase text-blue-700">{tx(locale, "Next Safe Commands", "安全命令")}</div>
                    {scientistUpgradePlan?.next_safe_commands?.slice(0, 4).map((command) => (
                      <div key={command} className="break-all font-mono text-[11px] text-blue-800">{command}</div>
                    ))}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Turn Plan", "科学家本轮计划")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "A per-turn control plan: intent, tool choices, rationale, gates, expected artifacts, and stop conditions before the agent answers or acts.",
                "每个回合先生成控制计划：识别意图、选择工具、解释理由、列出门禁、预期证据和停止条件，再回答或行动。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistTurnPlan()} disabled={scientistTurnPlanBusy}>
            {scientistTurnPlanBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <BrainCircuit className="h-4 w-4" />}
            {tx(locale, "Build Turn Plan", "生成本轮计划")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.7fr_1.3fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Intent", "意图")} value={<StatusBadge tone={turnPlanIntent === "execution" ? "amber" : turnPlanIntent === "not_run" ? "slate" : "blue"}>{turnPlanIntent}</StatusBadge>} />
            <Row label={tx(locale, "Payload", "载荷")} value={turnPlanPayload || "(none)"} />
            <Row label={tx(locale, "Autonomy", "自主级别")} value={scientistTurnPlan?.autonomy_level ?? "not_run"} />
            <Row label={tx(locale, "Can Execute", "可执行")} value={String(turnPlanReadiness.can_execute ?? false)} />
            <Row label={tx(locale, "Compute", "算力")} value={String(turnPlanReadiness.compute_backend ?? "unknown")} />
            <Row label={tx(locale, "Tool Count", "工具数")} value={turnPlanTools.length} />
            <Row label={tx(locale, "Next Safe", "安全下一步")} value={scientistTurnPlan?.next_safe_command ?? "evomind turn-plan"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistTurnPlan?.artifact_path ?? ".xsci/scientist_turn_plan.json"} />
          </div>
          <div className="grid gap-3 xl:grid-cols-3">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Selected Tools", "已选工具")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {turnPlanTools.length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No turn plan yet.", "尚未生成本轮计划。")}</div>
                ) : null}
                {turnPlanTools.slice(0, 8).map((item, index) => (
                  <div key={`${item.tool ?? "tool"}-${index}`} className="rounded border border-blue-100 bg-blue-50 px-2 py-1.5 text-xs">
                    <div className="flex items-start justify-between gap-2">
                      <span className="break-all font-mono font-bold text-blue-900">{item.tool ?? "unknown_tool"}</span>
                      <StatusBadge tone={item.gate === "read_only" ? "green" : item.gate?.includes("gate") ? "amber" : "slate"}>{item.confidence ?? "n/a"}</StatusBadge>
                    </div>
                    <div className="mt-1 leading-4 text-blue-800">{item.why ?? ""}</div>
                    {item.gate ? <div className="mt-1 font-mono text-[11px] text-blue-700">gate={item.gate}</div> : null}
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Stop Conditions", "停止条件")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {turnPlanStopConditions.length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No stop conditions yet.", "尚未生成停止条件。")}</div>
                ) : null}
                {turnPlanStopConditions.slice(0, 6).map((item, index) => (
                  <div key={`${item}-${index}`} className="rounded border border-amber-100 bg-amber-50 px-2 py-1.5 text-xs leading-4 text-amber-800">
                    {item}
                  </div>
                ))}
                {turnPlanBlockingGates.length ? (
                  <div className="rounded border border-red-100 bg-red-50 px-2 py-1.5 text-xs leading-4 text-red-800">
                    <div className="mb-1 font-bold">{tx(locale, "Blocking Gates", "阻塞门禁")}</div>
                    {turnPlanBlockingGates.slice(0, 3).map((item, index) => (
                      <div key={`${item}-${index}`}>{item}</div>
                    ))}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Evidence", "证据")}</div>
              <div className="max-h-64 space-y-1 overflow-y-auto pr-1">
                {turnPlanExpectedArtifacts.length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No expected artifacts yet.", "尚未生成预期证据。")}</div>
                ) : null}
                {turnPlanExpectedArtifacts.slice(0, 8).map((item, index) => (
                  <div key={`${item}-${index}`} className="break-all rounded bg-slate-50 px-2 py-1 font-mono text-[11px] text-slate-700">
                    {item}
                  </div>
                ))}
                {turnPlanAdvisoryGaps.length ? (
                  <div className="mt-2 rounded border border-slate-200 bg-slate-50 p-2 text-xs leading-4 text-slate-600">
                    <div className="mb-1 font-bold">{tx(locale, "Advisory Gaps", "提示缺口")}</div>
                    {turnPlanAdvisoryGaps.slice(0, 3).map((item, index) => (
                      <div key={`${item}-${index}`}>{item}</div>
                    ))}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Situation Model", "科学家局势模型")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "The central reasoning snapshot: evidence, uncertainty, blockers, memory, strategy, and the next safe tool sequence.",
                "核心推理快照：综合证据、不确定性、阻塞项、记忆、策略和下一步安全工具序列。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistSituationModel()} disabled={scientistSituationModelBusy}>
            {scientistSituationModelBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <BrainCircuit className="h-4 w-4" />}
            {tx(locale, "Analyze Situation", "分析当前局势")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.76fr_1.24fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Status", "状态")}
              value={<StatusBadge tone={situationReadiness >= 80 ? "green" : situationReadiness >= 55 ? "amber" : "red"}>{scientistSituationModel?.situation_status ?? "not_run"}</StatusBadge>}
            />
            <Row
              label={tx(locale, "Readiness", "就绪分")}
              value={<StatusBadge tone={situationReadiness >= 80 ? "green" : situationReadiness >= 55 ? "amber" : "red"}>{situationReadiness}</StatusBadge>}
            />
            <Row label={tx(locale, "Task", "任务")} value={scientistSituationModel?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Posture", "行动姿态")} value={situationBody?.posture ?? "not_run"} />
            <Row label={tx(locale, "Passed Checks", "通过检查")} value={situationPassedChecks.length} />
            <Row label={tx(locale, "Missing Checks", "缺失检查")} value={situationMissingChecks.length} />
            <Row label={tx(locale, "Training", "训练")} value={scientistSituationModel?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistSituationModel?.official_submit ?? "blocked_until_explicit_human_approval"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistSituationModel?.artifact_path ?? ".xsci/scientist_situation_model.json"} />
          </div>
          <div className="grid gap-3 xl:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Research Question", "研究问题")}</div>
              <p className="text-xs leading-5 text-slate-700">
                {situationBody?.research_question ?? tx(locale, "Run the situation model to synthesize the current research state.", "运行局势模型后，会综合当前研究状态。")}
              </p>
              {(situationNextCommands ?? []).length ? (
                <div className="mt-3 rounded border border-blue-100 bg-blue-50 p-2">
                  <div className="mb-1 text-xs font-bold uppercase text-blue-700">{tx(locale, "Next Safe Sequence", "下一步安全序列")}</div>
                  <div className="space-y-1">
                    {situationNextCommands.slice(0, 5).map((command, index) => (
                      <div key={`${command}-${index}`} className="break-all font-mono text-[11px] font-semibold text-blue-800">{command}</div>
                    ))}
                  </div>
                </div>
              ) : null}
              {situationMissingChecks.length ? (
                <div className="mt-3 rounded border border-amber-100 bg-amber-50 p-2">
                  <div className="mb-1 text-xs font-bold uppercase text-amber-700">{tx(locale, "Missing Readiness", "缺失就绪项")}</div>
                  <div className="flex flex-wrap gap-1">
                    {situationMissingChecks.slice(0, 8).map((item) => (
                      <span key={item} className="rounded bg-white px-1.5 py-0.5 font-mono text-[11px] text-amber-800">{item}</span>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Blockers & Uncertainty", "阻塞与不确定性")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {situationBlockers.length === 0 && situationUncertainties.length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No blocker model yet.", "暂无阻塞模型。")}</div>
                ) : null}
                {situationBlockers.slice(0, 5).map((item, index) => (
                  <div key={`${item.category ?? "blocker"}-${index}`} className="rounded border border-red-100 bg-red-50 px-2 py-1.5 text-xs">
                    <div className="flex items-start justify-between gap-2">
                      <span className="font-bold text-red-900">{item.category ?? "blocker"}</span>
                      <StatusBadge tone={item.severity === "high" ? "red" : item.severity === "medium" ? "amber" : "slate"}>{item.severity ?? "unknown"}</StatusBadge>
                    </div>
                    <div className="mt-1 leading-4 text-red-800">{item.blocker ?? ""}</div>
                    {item.repair_command ? <div className="mt-1 font-mono text-[11px] text-red-700">{item.repair_command}</div> : null}
                  </div>
                ))}
                {situationUncertainties.slice(0, 5).map((item, index) => (
                  <div key={`${item}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs leading-4 text-slate-600">
                    {item}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Readiness Report", "科学家就绪报告")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Unified go/no-go view for capability, execution gates, claim boundaries, artifacts, and next safe commands.",
                "统一展示能力、执行门禁、声明边界、证据文件和下一步安全命令。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistReadinessReport()} disabled={scientistReadinessReportBusy}>
            {scientistReadinessReportBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <FileCheck2 className="h-4 w-4" />}
            {tx(locale, "Run Report", "生成报告")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.75fr_1.25fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Score", "评分")}
              value={<StatusBadge tone={(scientistReadinessReport?.overall_score ?? 0) >= 85 ? "green" : (scientistReadinessReport?.overall_score ?? 0) >= 70 ? "blue" : (scientistReadinessReport?.overall_score ?? 0) >= 50 ? "amber" : "red"}>{scientistReadinessReport?.overall_score ?? 0}</StatusBadge>}
            />
            <Row label={tx(locale, "Capability", "能力")} value={<StatusBadge tone={readinessTone(scientistReadinessReport?.capability_readiness)}>{scientistReadinessReport?.capability_readiness ?? "not_run"}</StatusBadge>} />
            <Row label={tx(locale, "Launch", "上线")} value={<StatusBadge tone={readinessTone(scientistReadinessReport?.launch_readiness)}>{scientistReadinessReport?.launch_readiness ?? "not_run"}</StatusBadge>} />
            <Row label={tx(locale, "Training Claim", "训练声明")} value={String(scientistReadinessReport?.claim_readiness?.training_readiness_claim ?? "not_run")} />
            <Row label={tx(locale, "Rank Claim", "排名声明")} value={String(scientistReadinessReport?.claim_readiness?.rank_or_medal_claim ?? "blocked_without_kaggle_response_artifact")} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistReadinessReport?.official_submit ?? "blocked_until_explicit_human_approval"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistReadinessReport?.artifact_path ?? ".xsci/scientist_readiness_report.json"} />
            <Row label={tx(locale, "Markdown", "Markdown")} value={scientistReadinessReport?.markdown_artifact_path ?? ".xsci/scientist_readiness_report.md"} />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Gate Matrix", "门禁矩阵")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {(scientistReadinessReport?.readiness_matrix ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "Run readiness report to see gates.", "生成就绪报告后显示门禁。")}</div>
                ) : (
                  scientistReadinessReport?.readiness_matrix?.slice(0, 8).map((item, index) => (
                    <div key={`${item.name ?? "gate"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-bold text-slate-800">{item.name ?? "gate"}</span>
                        <StatusBadge tone={item.ok ? "green" : "red"}>{item.status ?? "unknown"}</StatusBadge>
                      </div>
                      <div className="mt-1 leading-4 text-slate-500">{item.evidence ?? ""}</div>
                      {item.next_action ? <div className="mt-1 font-mono text-[11px] text-slate-700">{item.next_action}</div> : null}
                    </div>
                  ))
                )}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Next Commands", "下一步命令")}</div>
              <div className="space-y-2">
                {(scientistReadinessReport?.recommended_next_commands ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No report commands yet.", "暂无报告命令。")}</div>
                ) : (
                  scientistReadinessReport?.recommended_next_commands?.slice(0, 6).map((command, index) => (
                    <div key={`${command}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 font-mono text-[11px] text-slate-700">
                      {command}
                    </div>
                  ))
                )}
              </div>
              <div className="mt-3 border-t border-slate-100 pt-2">
                <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Blocking Reasons", "阻塞原因")}</div>
                <div className="space-y-1.5">
                  {(scientistReadinessReport?.blocking_reasons ?? []).length === 0 ? (
                    <div className="text-xs text-slate-400">{tx(locale, "No blockers recorded.", "暂无阻塞记录。")}</div>
                  ) : (
                    scientistReadinessReport?.blocking_reasons?.slice(0, 4).map((item, index) => (
                      <div key={`${item}-${index}`} className="rounded border border-red-100 bg-red-50 px-2 py-1.5 text-xs leading-4 text-red-800">{item}</div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Causal Diagnosis", "科学家因果诊断")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Explains why the system is blocked or ready by linking symptoms, root causes, evidence, and safe interventions.",
                "把症状、根因、证据和安全干预动作串成可复盘因果图，解释系统为什么卡住或为什么就绪。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistCausalDiagnosis()} disabled={scientistCausalDiagnosisBusy}>
            {scientistCausalDiagnosisBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <GitBranch className="h-4 w-4" />}
            {tx(locale, "Run Diagnosis", "运行诊断")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.72fr_1.28fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Posture", "姿态")} value={<StatusBadge tone={readinessTone(scientistCausalDiagnosis?.posture)}>{scientistCausalDiagnosis?.posture ?? "not_run"}</StatusBadge>} />
            <Row label={tx(locale, "Symptoms", "症状")} value={scientistCausalDiagnosis?.symptoms?.length ?? 0} />
            <Row label={tx(locale, "Root Causes", "根因")} value={scientistCausalDiagnosis?.root_causes?.length ?? 0} />
            <Row label={tx(locale, "Interventions", "干预")} value={scientistCausalDiagnosis?.interventions?.length ?? 0} />
            <Row label={tx(locale, "Next", "下一步")} value={scientistCausalDiagnosis?.next_safe_command ?? "evomind causal-diagnosis"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistCausalDiagnosis?.artifact_path ?? ".xsci/scientist_causal_diagnosis.json"} />
            <Row label={tx(locale, "Markdown", "Markdown")} value={scientistCausalDiagnosis?.markdown_artifact_path ?? ".xsci/scientist_causal_diagnosis.md"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistCausalDiagnosis?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistCausalDiagnosis?.official_submit ?? "blocked_until_explicit_human_approval"} />
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Symptoms", "症状")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {(scientistCausalDiagnosis?.symptoms ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "Run causal diagnosis to see symptoms.", "运行因果诊断后显示症状。")}</div>
                ) : (
                  scientistCausalDiagnosis?.symptoms?.slice(0, 6).map((item, index) => (
                    <div key={`${item.id ?? "symptom"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-bold text-slate-800">{item.id ?? "symptom"}</span>
                        <StatusBadge tone={item.severity === "high" ? "red" : item.severity === "medium" ? "amber" : "slate"}>{item.severity ?? "unknown"}</StatusBadge>
                      </div>
                      <div className="mt-1 leading-4 text-slate-500">{item.summary ?? ""}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Root Causes", "根因")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {(scientistCausalDiagnosis?.root_causes ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No root causes yet.", "暂无根因。")}</div>
                ) : (
                  scientistCausalDiagnosis?.root_causes?.slice(0, 6).map((item, index) => (
                    <div key={`${item.id ?? "cause"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs">
                      <div className="font-bold text-slate-800">{item.id ?? "root_cause"}</div>
                      <div className="mt-1 flex items-center gap-2 text-[11px] text-slate-500">
                        <span>confidence={item.confidence ?? "n/a"}</span>
                        <span>{item.gate ?? "gate"}</span>
                      </div>
                      <div className="mt-1 leading-4 text-slate-500">{item.summary ?? ""}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Interventions", "干预动作")}</div>
              <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
                {(scientistCausalDiagnosis?.interventions ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No interventions yet.", "暂无干预动作。")}</div>
                ) : (
                  scientistCausalDiagnosis?.interventions?.slice(0, 6).map((item, index) => (
                    <div key={`${item.id ?? "intervention"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs">
                      <div className="font-bold text-slate-800">{item.title ?? item.id ?? "intervention"}</div>
                      <div className="mt-1 font-mono text-[11px] text-slate-700">{item.safe_next_command ?? ""}</div>
                      <div className="mt-1 text-[11px] text-slate-500">{item.gate ?? "gate"}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Self-Audit", "科学家自我审计")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Read-only capability audit of EvoMind itself: scores, gaps, evidence sources, and system-upgrade backlog.",
                "只读审计 EvoMind 自身能力：评分、缺口、证据来源和系统升级 backlog。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistSelfAudit()} disabled={scientistSelfAuditBusy}>
            {scientistSelfAuditBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
            {tx(locale, "Run Self-Audit", "运行自我审计")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.78fr_1.22fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Score", "评分")}
              value={<StatusBadge tone={(scientistSelfAudit?.overall_score ?? 0) >= 85 ? "green" : (scientistSelfAudit?.overall_score ?? 0) >= 70 ? "blue" : (scientistSelfAudit?.overall_score ?? 0) >= 50 ? "amber" : "red"}>{scientistSelfAudit?.overall_score ?? 0}</StatusBadge>}
            />
            <Row label={tx(locale, "Capability", "能力状态")} value={scientistSelfAudit?.capability_readiness ?? "not_run"} />
            <Row label={tx(locale, "Launch Readiness", "上线状态")} value={scientistSelfAudit?.launch_readiness ?? "not_run"} />
            <Row
              label={tx(locale, "Training Claim", "训练声明")}
              value={String(scientistSelfAudit?.claim_readiness?.training_readiness_claim ?? "not_run")}
            />
            <Row
              label={tx(locale, "Rank Claim", "排名声明")}
              value={String(scientistSelfAudit?.claim_readiness?.rank_or_medal_claim ?? "blocked_without_kaggle_response_artifact")}
            />
            <Row label={tx(locale, "Task", "任务")} value={scientistSelfAudit?.selected_task || selectedTask || "(none)"} />
            <Row
              label={tx(locale, "Trend", "能力趋势")}
              value={
                scientistSelfAudit?.capability_trend ? (
                  <span className="font-mono text-[11px]">
                    {scientistSelfAudit.capability_trend.previous_score ?? "first"} → {scientistSelfAudit.capability_trend.current_score ?? scientistSelfAudit.overall_score ?? 0}
                    {scientistSelfAudit.capability_trend.score_delta == null ? "" : ` (${scientistSelfAudit.capability_trend.score_delta >= 0 ? "+" : ""}${scientistSelfAudit.capability_trend.score_delta})`}
                  </span>
                ) : "not_run"
              }
            />
            <Row label={tx(locale, "Trend Records", "趋势记录")} value={scientistSelfAudit?.capability_trend?.records_after ?? 0} />
            <Row label={tx(locale, "Gaps", "缺口")} value={scientistSelfAudit?.gaps?.length ?? 0} />
            <Row label={tx(locale, "Backlog", "升级项")} value={scientistSelfAudit?.upgrade_backlog?.length ?? 0} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistSelfAudit?.artifact_path ?? ".xsci/scientist_self_audit.json"} />
            <Row label={tx(locale, "Backlog File", "升级文件")} value={scientistSelfAudit?.backlog_artifact_path ?? ".xsci/scientist_upgrade_backlog.json"} />
            <Row label={tx(locale, "Trend File", "趋势文件")} value={scientistSelfAudit?.capability_trend?.path ?? ".xsci/scientist_capability_trend.jsonl"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistSelfAudit?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistSelfAudit?.official_submit ?? "blocked_until_explicit_human_approval"} />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Capability Scores", "能力评分")}</div>
              <div className="max-h-56 space-y-2 overflow-y-auto pr-1">
                {(scientistSelfAudit?.capabilities ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "Run self-audit to see capability scores.", "运行自我审计后显示能力评分。")}</div>
                ) : (
                  scientistSelfAudit?.capabilities?.slice(0, 8).map((item, index) => (
                    <div key={`${item.name ?? "cap"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1.5 text-xs">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-bold text-slate-800">{item.name ?? "capability"}</span>
                        <StatusBadge tone={(item.score ?? 0) >= 85 ? "green" : (item.score ?? 0) >= 70 ? "blue" : (item.score ?? 0) >= 50 ? "amber" : "red"}>{item.score ?? 0}</StatusBadge>
                      </div>
                      <div className="mt-1 text-[11px] text-slate-500">{item.status ?? "unknown"}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Upgrade Backlog", "升级 Backlog")}</div>
              <div className="max-h-56 space-y-2 overflow-y-auto pr-1">
                {(scientistSelfAudit?.upgrade_backlog ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No backlog yet.", "暂无升级项。")}</div>
                ) : (
                  scientistSelfAudit?.upgrade_backlog?.slice(0, 6).map((item, index) => (
                    <div key={`${item.id ?? "item"}-${index}`} className="rounded border border-slate-100 bg-white px-2 py-1.5 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <span className="font-bold text-slate-800">{item.title ?? item.id ?? "upgrade"}</span>
                        <StatusBadge tone={item.priority === "P0" ? "red" : item.priority === "P1" ? "amber" : "slate"}>{item.priority ?? "P?"}</StatusBadge>
                      </div>
                      {item.safe_next_command ? <div className="mt-1 font-mono text-[11px] text-blue-700">{item.safe_next_command}</div> : null}
                      {item.why ? <div className="mt-1 text-[11px] leading-4 text-slate-500">{item.why.slice(0, 160)}</div> : null}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Experiment Blueprint", "实验蓝图")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Execution design layer: converts the reviewed hypothesis into branch, code mode, resource mode, artifacts, gates, rollback, and memory writeback before training.",
                "执行设计层：把已评审假设转成训练前可审计的分支、代码模式、资源模式、证据文件、门禁、回滚条件和记忆写回计划。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistExperimentBlueprint()} disabled={scientistExperimentBlueprintBusy}>
            {scientistExperimentBlueprintBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <GitBranch className="h-4 w-4" />}
            {tx(locale, "Build Blueprint", "生成实验蓝图")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.78fr_1.22fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Status", "状态")}
              value={<StatusBadge tone={scientistExperimentBlueprint?.blueprint_status === "ready_for_gated_execution" ? "green" : scientistExperimentBlueprint?.blueprint_status === "blocked_until_gates_clear" ? "amber" : "slate"}>{scientistExperimentBlueprint?.blueprint_status ?? "not_run"}</StatusBadge>}
            />
            <Row label={tx(locale, "Task", "任务")} value={scientistExperimentBlueprint?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistExperimentBlueprint?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistExperimentBlueprint?.official_submit ?? "blocked_until_explicit_human_approval"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistExperimentBlueprint?.artifact_path ?? ".xsci/scientist_experiment_blueprint.json"} />
            <Row label={tx(locale, "Source Review", "来源评审")} value={scientistExperimentBlueprint?.source_review_path ?? ".xsci/scientist_hypothesis_review.json"} />
            <div className="mt-3 rounded border border-slate-200 bg-white p-2">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Gate Summary", "门禁摘要")}</div>
              <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600">
                {Object.entries(scientistExperimentBlueprint?.gate_summary ?? {}).slice(0, 8).map(([key, value]) => (
                  <div key={key} className="rounded bg-slate-50 px-2 py-1">
                    <div className="font-mono text-slate-500">{key}</div>
                    <div className="break-all font-bold text-slate-900">{String(value ?? "")}</div>
                  </div>
                ))}
                {Object.entries(scientistExperimentBlueprint?.gate_summary ?? {}).length === 0 ? (
                  <div className="col-span-2 text-xs text-slate-400">{tx(locale, "Build a blueprint to see contract, data, and evidence gates.", "生成实验蓝图后显示契约、数据和证据门禁。")}</div>
                ) : null}
              </div>
            </div>
            {(scientistExperimentBlueprint?.next_safe_commands ?? []).length ? (
              <div className="mt-3 rounded border border-blue-100 bg-blue-50 p-2">
                <div className="mb-1 text-xs font-bold uppercase text-blue-700">{tx(locale, "Next Safe Commands", "安全下一步命令")}</div>
                <div className="space-y-1">
                  {scientistExperimentBlueprint?.next_safe_commands?.slice(0, 4).map((command, index) => (
                    <div key={`${command}-${index}`} className="break-all font-mono text-[11px] font-semibold text-blue-800">{command}</div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
          <div className="space-y-3">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-xs font-bold uppercase text-slate-500">{tx(locale, "Selected Hypothesis", "选中假设")}</div>
                <StatusBadge tone={scientistExperimentBlueprint?.selected_hypothesis ? "blue" : "slate"}>
                  {scientistExperimentBlueprint?.selected_hypothesis?.hypothesis_id ?? "not_selected"}
                </StatusBadge>
              </div>
              {!scientistExperimentBlueprint?.selected_hypothesis ? (
                <div className="text-xs text-slate-400">{tx(locale, "Run hypothesis review, then build the blueprint.", "先运行假设评审，再生成实验蓝图。")}</div>
              ) : (
                <div className="space-y-2 text-xs text-slate-700">
                  <div className="font-bold text-slate-950">{scientistExperimentBlueprint.selected_hypothesis.strategy_name ?? scientistExperimentBlueprint.selected_hypothesis.hypothesis_id}</div>
                  <div className="font-mono text-[11px] text-slate-500">
                    {scientistExperimentBlueprint.selected_hypothesis.branch_type ?? "branch"} | {scientistExperimentBlueprint.selected_hypothesis.code_generation_mode ?? "stepwise"}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="rounded border border-slate-100 bg-slate-50 p-2">
                      <div className="font-bold uppercase text-slate-500">Score</div>
                      <div className="mt-1 font-bold text-slate-900">{scientistExperimentBlueprint.selected_hypothesis.score ?? 0}</div>
                    </div>
                    <div className="rounded border border-slate-100 bg-slate-50 p-2">
                      <div className="font-bold uppercase text-slate-500">Risk</div>
                      <div className="mt-1 font-bold text-slate-900">{scientistExperimentBlueprint.selected_hypothesis.risk_level ?? "unknown"}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Execution Blueprint", "执行蓝图")}</div>
              {!scientistExperimentBlueprint?.experiment_blueprint ? (
                <div className="text-xs text-slate-400">{tx(locale, "No experiment blueprint yet.", "暂无实验蓝图。")}</div>
              ) : (
                <div className="space-y-2 text-xs text-slate-700">
                  <Row label="Blueprint ID" value={scientistExperimentBlueprint.experiment_blueprint.blueprint_id ?? "n/a"} />
                  <Row label={tx(locale, "Branch", "分支")} value={scientistExperimentBlueprint.experiment_blueprint.branch_type ?? "n/a"} />
                  <Row label={tx(locale, "Code Mode", "代码模式")} value={scientistExperimentBlueprint.experiment_blueprint.code_generation_mode ?? "n/a"} />
                  <Row label={tx(locale, "Resource", "资源模式")} value={scientistExperimentBlueprint.experiment_blueprint.resource_mode ?? "n/a"} />
                  <Row label={tx(locale, "Run Command", "运行命令")} value={<span className="break-all font-mono text-[11px] text-blue-700">{scientistExperimentBlueprint.experiment_blueprint.run_command ?? "n/a"}</span>} />
                  <Row label={tx(locale, "Dry Run", "干运行")} value={<span className="break-all font-mono text-[11px] text-blue-700">{scientistExperimentBlueprint.experiment_blueprint.dry_run_command ?? "n/a"}</span>} />
                  <Row label={tx(locale, "Rollback", "回滚条件")} value={scientistExperimentBlueprint.experiment_blueprint.rollback_condition ?? "hold if gates fail"} />
                  <Row label={tx(locale, "Claim Boundary", "声明边界")} value={scientistExperimentBlueprint.experiment_blueprint.claim_boundary ?? "official claims blocked without Kaggle response"} />
                </div>
              )}
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-md border border-slate-200 bg-white p-3">
                <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Required Artifacts", "必需证据")}</div>
                <div className="max-h-44 space-y-1 overflow-y-auto pr-1">
                  {(scientistExperimentBlueprint?.experiment_blueprint?.required_artifacts ?? []).length === 0 ? (
                    <div className="text-xs text-slate-400">{tx(locale, "No artifact contract yet.", "暂无证据契约。")}</div>
                  ) : (
                    scientistExperimentBlueprint?.experiment_blueprint?.required_artifacts?.slice(0, 10).map((item, index) => (
                      <div key={`${item}-${index}`} className="break-all rounded border border-slate-100 bg-slate-50 px-2 py-1 font-mono text-[11px] text-slate-700">- {item}</div>
                    ))
                  )}
                </div>
              </div>
              <div className="rounded-md border border-slate-200 bg-white p-3">
                <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Promotion Gates", "晋升门禁")}</div>
                <div className="max-h-44 space-y-1 overflow-y-auto pr-1">
                  {(scientistExperimentBlueprint?.experiment_blueprint?.promotion_gates ?? []).length === 0 ? (
                    <div className="text-xs text-slate-400">{tx(locale, "No promotion gate contract yet.", "暂无晋升门禁契约。")}</div>
                  ) : (
                    scientistExperimentBlueprint?.experiment_blueprint?.promotion_gates?.slice(0, 10).map((item, index) => (
                      <div key={`${item}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1 text-[11px] font-semibold text-slate-700">- {item}</div>
                    ))
                  )}
                </div>
              </div>
            </div>
            {scientistExperimentBlueprint?.experiment_blueprint?.memory_writeback_plan ? (
              <div className="rounded-md border border-blue-100 bg-blue-50 p-3 text-xs text-blue-900">
                <div className="mb-2 font-bold uppercase text-blue-700">{tx(locale, "Memory Writeback", "记忆写回")}</div>
                <JsonInspector data={scientistExperimentBlueprint.experiment_blueprint.memory_writeback_plan} />
              </div>
            ) : null}
            {(scientistExperimentBlueprint?.experiment_blueprint?.memory_reuse_plan ?? scientistExperimentBlueprint?.memory_reuse_plan) ? (
              <div className="rounded-md border border-emerald-100 bg-emerald-50 p-3 text-xs text-emerald-950">
                <div className="mb-2 font-bold uppercase text-emerald-700">{tx(locale, "Memory Reuse", "记忆复用")}</div>
                <JsonInspector data={scientistExperimentBlueprint.experiment_blueprint?.memory_reuse_plan ?? scientistExperimentBlueprint.memory_reuse_plan ?? {}} />
              </div>
            ) : null}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Hypothesis Review", "假设评审")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Scientist critique layer: rank proposed branches by evidence, data readiness, expected impact, risk, and promotion gates before any training.",
                "科学家评审层：训练前按证据、数据就绪度、预期影响、风险和晋升门禁评审候选分支。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistHypothesisReview()} disabled={scientistHypothesisReviewBusy}>
            {scientistHypothesisReviewBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <FileCheck2 className="h-4 w-4" />}
            {tx(locale, "Review Hypotheses", "评审假设")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.78fr_1.22fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Recommendation", "建议")} value={scientistHypothesisReview?.recommendation ?? "not_run"} />
            <Row label={tx(locale, "Task", "任务")} value={scientistHypothesisReview?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Reviewed", "已评审")} value={scientistHypothesisReview?.hypotheses_reviewed ?? scientistHypothesisReview?.reviews?.length ?? 0} />
            <Row label={tx(locale, "Training", "训练")} value={scientistHypothesisReview?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistHypothesisReview?.official_submit ?? "blocked_until_explicit_human_approval"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistHypothesisReview?.artifact_path ?? ".xsci/scientist_hypothesis_review.json"} />
            <Row label={tx(locale, "Source", "来源")} value={scientistHypothesisReview?.source_backlog_path ?? ".xsci/scientist_innovation_backlog.json"} />
            <div className="mt-3 rounded border border-slate-200 bg-white p-2">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Gate Summary", "门禁摘要")}</div>
              <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600">
                {Object.entries(scientistHypothesisReview?.gate_summary ?? {}).slice(0, 8).map(([key, value]) => (
                  <div key={key} className="rounded bg-slate-50 px-2 py-1">
                    <div className="font-mono text-slate-500">{key}</div>
                    <div className="break-all font-bold text-slate-900">{String(value ?? "")}</div>
                  </div>
                ))}
                {Object.entries(scientistHypothesisReview?.gate_summary ?? {}).length === 0 ? (
                  <div className="col-span-2 text-xs text-slate-400">{tx(locale, "Run review to see data, memory, and contract gates.", "运行评审后显示数据、记忆和执行契约门禁。")}</div>
                ) : null}
              </div>
            </div>
            {(scientistHypothesisReview?.next_safe_commands ?? []).length ? (
              <div className="mt-3 rounded border border-blue-100 bg-blue-50 p-2">
                <div className="mb-1 text-xs font-bold uppercase text-blue-700">{tx(locale, "Next Safe Commands", "安全下一步命令")}</div>
                <div className="space-y-1">
                  {scientistHypothesisReview?.next_safe_commands?.slice(0, 4).map((command, index) => (
                    <div key={`${command}-${index}`} className="break-all font-mono text-[11px] font-semibold text-blue-800">{command}</div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
          <div className="space-y-3">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-xs font-bold uppercase text-slate-500">{tx(locale, "Selected Hypothesis", "选中假设")}</div>
                <StatusBadge tone={scientistHypothesisReview?.selected_hypothesis?.status === "ready_for_gated_execution" ? "green" : scientistHypothesisReview?.selected_hypothesis ? "amber" : "slate"}>
                  {scientistHypothesisReview?.selected_hypothesis?.status ?? "not_selected"}
                </StatusBadge>
              </div>
              {!scientistHypothesisReview?.selected_hypothesis ? (
                <div className="text-xs text-slate-400">{tx(locale, "No reviewed hypothesis yet.", "暂无已评审假设。")}</div>
              ) : (
                <div className="space-y-2 text-xs text-slate-700">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="font-bold text-slate-950">{scientistHypothesisReview.selected_hypothesis.strategy_name ?? scientistHypothesisReview.selected_hypothesis.hypothesis_id}</div>
                      <div className="mt-1 font-mono text-[11px] text-slate-500">
                        {scientistHypothesisReview.selected_hypothesis.branch_type ?? "branch"} | {scientistHypothesisReview.selected_hypothesis.code_generation_mode ?? "stepwise"}
                      </div>
                    </div>
                    <StatusBadge tone={scientistHypothesisReview.selected_hypothesis.risk_level === "low" ? "green" : scientistHypothesisReview.selected_hypothesis.risk_level === "high" ? "red" : "amber"}>
                      score={scientistHypothesisReview.selected_hypothesis.score ?? 0}
                    </StatusBadge>
                  </div>
                  <div className="grid gap-2 md:grid-cols-4">
                    <div className="rounded border border-slate-100 bg-slate-50 p-2">
                      <div className="font-bold uppercase text-slate-500">Evidence</div>
                      <div className="mt-1 font-bold text-slate-900">{scientistHypothesisReview.selected_hypothesis.evidence_score ?? 0}</div>
                    </div>
                    <div className="rounded border border-slate-100 bg-slate-50 p-2">
                      <div className="font-bold uppercase text-slate-500">Ready</div>
                      <div className="mt-1 font-bold text-slate-900">{scientistHypothesisReview.selected_hypothesis.readiness_score ?? 0}</div>
                    </div>
                    <div className="rounded border border-slate-100 bg-slate-50 p-2">
                      <div className="font-bold uppercase text-slate-500">Impact</div>
                      <div className="mt-1 font-bold text-slate-900">{scientistHypothesisReview.selected_hypothesis.impact_score ?? 0}</div>
                    </div>
                    <div className="rounded border border-slate-100 bg-slate-50 p-2">
                      <div className="font-bold uppercase text-slate-500">Risk</div>
                      <div className="mt-1 font-bold text-slate-900">{scientistHypothesisReview.selected_hypothesis.risk_penalty ?? 0}</div>
                    </div>
                  </div>
                  <div><span className="font-bold text-slate-500">{tx(locale, "Next Gate", "下一门禁")}</span>: {scientistHypothesisReview.selected_hypothesis.next_gate ?? "score_promotion_gate"}</div>
                  {(scientistHypothesisReview.selected_hypothesis.reasons ?? []).length ? (
                    <div className="rounded border border-green-100 bg-green-50 p-2 text-green-800">
                      {(scientistHypothesisReview.selected_hypothesis.reasons ?? []).slice(0, 4).map((reason, index) => <div key={`${reason}-${index}`}>- {reason}</div>)}
                    </div>
                  ) : null}
                  {(scientistHypothesisReview.selected_hypothesis.blockers ?? []).length ? (
                    <div className="rounded border border-amber-100 bg-amber-50 p-2 text-amber-800">
                      {(scientistHypothesisReview.selected_hypothesis.blockers ?? []).slice(0, 4).map((blocker, index) => <div key={`${blocker}-${index}`}>- {blocker}</div>)}
                    </div>
                  ) : null}
                </div>
              )}
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Ranked Reviews", "排序评审")}</div>
              <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                {(scientistHypothesisReview?.reviews ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "Generate and review hypotheses to see ranked proposals.", "生成并评审假设后显示排序方案。")}</div>
                ) : (
                  scientistHypothesisReview?.reviews?.slice(0, 8).map((item, index) => (
                    <div key={`${item.hypothesis_id ?? "review"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-3 py-2 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="font-bold text-slate-900">#{item.rank ?? index + 1} {item.strategy_name ?? item.hypothesis_id ?? "hypothesis"}</div>
                          <div className="mt-1 font-mono text-[11px] text-slate-500">{item.branch_type ?? "branch"} | {item.code_generation_mode ?? "stepwise"}</div>
                        </div>
                        <StatusBadge tone={item.status === "ready_for_gated_execution" ? "green" : item.status === "blocked" ? "red" : "amber"}>
                          {item.score ?? 0}
                        </StatusBadge>
                      </div>
                      <div className="mt-2 flex flex-wrap gap-1 text-[11px]">
                        <span className="rounded border border-slate-200 bg-white px-1.5 py-0.5">evidence {item.evidence_score ?? 0}</span>
                        <span className="rounded border border-slate-200 bg-white px-1.5 py-0.5">ready {item.readiness_score ?? 0}</span>
                        <span className="rounded border border-slate-200 bg-white px-1.5 py-0.5">impact {item.impact_score ?? 0}</span>
                        <span className="rounded border border-slate-200 bg-white px-1.5 py-0.5">risk {item.risk_penalty ?? 0}</span>
                      </div>
                      {(item.reasons ?? []).length ? (
                        <div className="mt-2 leading-4 text-slate-600">{(item.reasons ?? []).slice(0, 2).join(" ")}</div>
                      ) : null}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Research Hypotheses", "创新假设")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Memory-guided proposal backlog before training: hypotheses, branch types, risk controls, and required artifacts.",
                "训练前的记忆复用提案：创新假设、分支类型、风险控制和必须产出的证据。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistInnovationBacklog()} disabled={scientistInnovationBusy}>
            {scientistInnovationBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <Lightbulb className="h-4 w-4" />}
            {tx(locale, "Generate Hypotheses", "生成创新假设")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.78fr_1.22fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Task", "任务")} value={scientistInnovationBacklog?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Hypotheses", "假设数量")} value={scientistInnovationBacklog?.innovation_hypotheses?.length ?? 0} />
            <Row label={tx(locale, "Training", "训练")} value={scientistInnovationBacklog?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistInnovationBacklog?.official_submit ?? "blocked_until_explicit_human_approval"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistInnovationBacklog?.artifact_path ?? ".xsci/scientist_innovation_backlog.json"} />
            <Row label={tx(locale, "Innovation Log", "创新日志")} value={scientistInnovationBacklog?.innovation_log_path ?? ".xsci/innovation_log.json"} />
            <div className="mt-3 rounded border border-slate-200 bg-white p-2">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Memory Reuse", "记忆复用")}</div>
              <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600">
                {Object.entries(scientistInnovationBacklog?.memory_summary ?? {}).slice(0, 8).map(([key, value]) => (
                  <div key={key} className="rounded bg-slate-50 px-2 py-1">
                    <div className="font-mono text-slate-500">{key}</div>
                    <div className="font-bold text-slate-900">{String(value ?? 0)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="rounded-md border border-slate-200 bg-white p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="text-xs font-bold uppercase text-slate-500">{tx(locale, "Proposal Branches", "候选分支")}</div>
              <StatusBadge tone="blue">{tx(locale, "proposal-only", "仅提案")}</StatusBadge>
            </div>
            <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
              {(scientistInnovationBacklog?.innovation_hypotheses ?? []).length === 0 ? (
                <div className="text-xs text-slate-400">{tx(locale, "Generate hypotheses to see memory-guided branches.", "生成创新假设后显示记忆驱动的候选分支。")}</div>
              ) : (
                scientistInnovationBacklog?.innovation_hypotheses?.slice(0, 6).map((item, index) => (
                  <div key={`${item.id ?? "hyp"}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-3 py-2 text-xs">
                    <div className="flex items-start justify-between gap-2">
                      <div className="font-bold text-slate-900">{item.strategy_name ?? item.id ?? "hypothesis"}</div>
                      <StatusBadge tone="slate">{item.proposed_branch_type ?? "branch"}</StatusBadge>
                    </div>
                    <div className="mt-1 text-[11px] text-slate-500">
                      {item.code_generation_mode ?? "stepwise"} | {item.gate ?? "proposal_gate"}
                    </div>
                    {item.components?.length ? (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {item.components.slice(0, 5).map((component) => (
                          <span key={component} className="rounded border border-blue-100 bg-blue-50 px-1.5 py-0.5 text-[11px] text-blue-700">{component}</span>
                        ))}
                      </div>
                    ) : null}
                    {item.rationale ? <div className="mt-2 leading-4 text-slate-600">{item.rationale.slice(0, 220)}</div> : null}
                  </div>
                ))
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Autopilot", "科学家诊断 Autopilot")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Read-only multi-tool diagnosis: status, data, recent run, memory, gates, and next decision.",
                "只读多工具诊断：系统状态、数据、最近实验、记忆、门禁和下一步决策。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="primary" onClick={() => void runScientistAutopilot()} disabled={autopilotBusy}>
            {autopilotBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <BrainCircuit className="h-4 w-4" />}
            {tx(locale, "Run Diagnosis", "运行诊断")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Mode", "模式")}
              value={<StatusBadge tone={autopilot?.mode === "ready_to_execute" ? "green" : autopilot?.present ? "amber" : "slate"}>{autopilot?.mode ?? "not_run"}</StatusBadge>}
            />
            <Row
              label={tx(locale, "Runner", "运行器")}
              value={
                <StatusBadge tone={scientistAutopilotStatus?.running ? "blue" : scientistAutopilotStatus?.status === "completed" ? "green" : scientistAutopilotStatus?.status === "failed" ? "red" : "slate"}>
                  {scientistAutopilotStatus?.status ?? "not_started"}
                </StatusBadge>
              }
            />
            <Row label="Run ID" value={scientistAutopilotStatus?.run_id ?? "(none)"} />
            <Row label="PID" value={scientistAutopilotStatus?.pid ?? "(none)"} />
            <Row label={tx(locale, "Task", "任务")} value={autopilot?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={autopilot?.artifact_path ?? ".xsci/scientist_autopilot.json"} />
            {autopilot?.blockers?.length ? (
              <div className="mt-3 rounded-md border border-red-100 bg-red-50 p-2 text-xs text-red-700">
                <div className="mb-1 font-bold">{tx(locale, "Blockers", "阻塞项")}</div>
                {autopilot.blockers.slice(0, 4).map((item, index) => <div key={`${item}-${index}`}>- {item}</div>)}
              </div>
            ) : null}
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Tool Trace", "工具轨迹")}</div>
              <div className="max-h-40 space-y-1 overflow-y-auto pr-1">
                {(autopilot?.tool_trace ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No diagnosis trace yet.", "暂无诊断轨迹。")}</div>
                ) : (
                  autopilot?.tool_trace?.slice(0, 8).map((item, index) => (
                    <div key={`${item.tool}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-2 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="truncate font-semibold text-slate-800">{item.tool}</div>
                          <div className="mt-0.5 text-[11px] text-slate-500">
                            {tx(locale, "confidence", "置信度")}={
                              typeof item.confidence === "number" ? item.confidence.toFixed(2) : "n/a"
                            }
                            {item.evidence_signal ? ` · ${item.evidence_signal}` : ""}
                          </div>
                        </div>
                        <StatusBadge tone={item.ok === false ? "red" : "green"}>{item.status ?? (item.ok === false ? "blocked" : "ok")}</StatusBadge>
                      </div>
                      {item.rationale ? (
                        <div className="mt-1.5 leading-4 text-slate-600">
                          {item.rationale.slice(0, 220)}
                        </div>
                      ) : null}
                    </div>
                  ))
                )}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Next Actions", "下一步")}</div>
              <div className="max-h-40 space-y-1 overflow-y-auto pr-1 text-xs text-slate-700">
                {(autopilot?.next_actions ?? []).length === 0 ? (
                  <div className="text-slate-400">{tx(locale, "Run diagnosis to get the next action.", "运行诊断以生成下一步建议。")}</div>
                ) : (
                  autopilot?.next_actions?.slice(0, 5).map((item, index) => <div key={`${item}-${index}`}>- {item}</div>)
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Autonomous Loop", "科学家自主循环")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Bounded loop: diagnose, execute safe read-only next steps, detect stagnation, escalate into planning artifacts, and write reusable lessons.",
                "有界循环：诊断、执行只读安全下一步、识别停滞、升级为计划/修复/契约证据，并写入可复用经验。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="primary" onClick={() => void runScientistLoop()} disabled={busy || autopilotBusy || scientistLoopBusy}>
            {scientistLoopBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <RefreshCcw className="h-4 w-4" />}
            {tx(locale, "Run Loop", "运行循环")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.82fr_1.18fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Mode", "模式")}
              value={<StatusBadge tone={scientistLoop?.present ? (scientistLoop?.stop_reason?.includes("gate") ? "amber" : "green") : "slate"}>{scientistLoop?.mode ?? "not_run"}</StatusBadge>}
            />
            <Row label={tx(locale, "Stop Reason", "停止原因")} value={scientistLoop?.stop_reason ?? "not_run"} />
            <Row label={tx(locale, "Task", "任务")} value={scientistLoop?.selected_task || selectedTask || "(none)"} />
            <Row label="Trace" value={scientistLoop?.trace_run_id ?? "(none)"} />
            <Row label={tx(locale, "Steps", "步骤数")} value={scientistLoop?.steps?.length ?? 0} />
            <Row label={tx(locale, "Lessons", "经验数")} value={scientistLoopLessons?.count ?? 0} />
            <Row
              label={tx(locale, "Memory Writeback", "长期记忆写回")}
              value={
                <StatusBadge tone={scientistMemoryConsolidation?.present ? (scientistMemoryConsolidation?.ok === false ? "amber" : "green") : "slate"}>
                  {scientistMemoryConsolidation?.present ? tx(locale, "synced", "已写回") : tx(locale, "not_run", "未运行")}
                </StatusBadge>
              }
            />
            <Row label={tx(locale, "Memory Added", "新增记忆")} value={scientistMemoryConsolidation?.records_added ?? scientistLoop?.memory_records_added ?? 0} />
            <Row label={tx(locale, "Memory Total", "记忆总数")} value={scientistMemoryConsolidation?.records_total ?? scientistLoop?.memory_records_total ?? 0} />
            <Row
              label={tx(locale, "Patch Memory", "补丁记忆")}
              value={
                <StatusBadge tone={scientistMemoryConsolidation?.source_counts?.patch_work_order_present ? "green" : "slate"}>
                  {scientistMemoryConsolidation?.source_counts?.patch_work_order_present ? tx(locale, "absorbed", "已吸收") : tx(locale, "not_seen", "未发现")}
                </StatusBadge>
              }
            />
            <Row label={tx(locale, "Patch Trials", "补丁试验")} value={String(scientistMemoryConsolidation?.source_counts?.patch_trials ?? 0)} />
            <Row
              label={tx(locale, "Continuation Memory", "续跑记忆")}
              value={
                <StatusBadge tone={scientistMemoryConsolidation?.source_counts?.continuation_resume_present ? "green" : "slate"}>
                  {scientistMemoryConsolidation?.source_counts?.continuation_resume_present ? tx(locale, "absorbed", "已吸收") : tx(locale, "not_seen", "未发现")}
                </StatusBadge>
              }
            />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistLoop?.artifact_path ?? ".xsci/scientist_loop.json"} />
            <Row label={tx(locale, "Lesson Log", "经验日志")} value={scientistLoop?.lessons_path ?? scientistLoopLessons?.artifact_path ?? ".xsci/scientist_loop_lessons.jsonl"} />
            <Row label={tx(locale, "Memory Artifact", "记忆证据")} value={scientistMemoryConsolidation?.artifact_path ?? scientistLoop?.memory_consolidation_artifact_path ?? ".xsci/scientist_memory_consolidation.json"} />
            <Row label={tx(locale, "Memory Store", "记忆库")} value={scientistMemoryConsolidation?.memory_path ?? scientistLoop?.memory_path ?? "experiments/evolution/retrospective_memory.json"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistLoop?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistLoop?.official_submit ?? "blocked_until_explicit_human_approval"} />
            <div className="mt-3 rounded-md border border-blue-100 bg-blue-50 p-2 text-xs leading-5 text-blue-800">
              {tx(
                locale,
                "Loop execution is read-only and gate-aware. It improves the next decision and memory, but does not spend compute or submit Kaggle results.",
                "循环执行是只读且受门禁约束的：它会改进下一步决策和记忆，但不会消耗训练算力，也不会提交 Kaggle。"
              )}
            </div>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Loop Steps", "循环步骤")}</div>
              <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                {(scientistLoop?.steps ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "Run Scientist Loop to see the autonomous step trace.", "运行科学家自主循环后会显示步骤轨迹。")}</div>
                ) : (
                  scientistLoop?.steps?.slice(0, 10).map((step, index) => (
                    <div key={`${String(step.step ?? step.tool ?? index)}-${index}`} className="rounded-md border border-slate-100 bg-slate-50/70 px-3 py-2 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="truncate font-bold text-slate-800">{String(step.step ?? step.tool ?? `step_${index + 1}`)}</div>
                          <div className="mt-1 font-mono text-[11px] text-blue-700">{String(step.tool ?? "tool")}</div>
                        </div>
                        <StatusBadge tone={String(step.status ?? "").includes("blocked") ? "amber" : String(step.status ?? "") === "ok" || String(step.status ?? "").includes("executed") ? "green" : "slate"}>{String(step.status ?? "unknown")}</StatusBadge>
                      </div>
                      {step.selected_action ? <div className="mt-2 text-slate-600"><span className="font-bold">Action</span>: {String(step.selected_action)}</div> : null}
                      {step.executed_tool ? <div className="mt-1 text-slate-600"><span className="font-bold">Executed</span>: {String(step.executed_tool)}</div> : null}
                      {step.gate ? <div className="mt-1 text-amber-700"><span className="font-bold">Gate</span>: {String(step.gate)}</div> : null}
                      {step.artifact_path ? <div className="mt-1 break-all font-mono text-[11px] text-slate-500">{String(step.artifact_path)}</div> : null}
                    </div>
                  ))
                )}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Final Decision", "最终决策")}</div>
              <div className="space-y-2 text-xs text-slate-700">
                <Row label={tx(locale, "Status", "状态")} value={String(loopFinalNext?.status ?? "not_run")} />
                <Row label={tx(locale, "Selected", "选中动作")} value={String(loopFinalSelected?.id ?? "(none)")} />
                <Row label={tx(locale, "Command", "命令")} value={<span className="font-mono text-[11px] text-blue-700">{String(loopFinalSelected?.command ?? "evomind loop")}</span>} />
                <Row label="Gate" value={String(loopFinalSelected?.gate ?? "read_only")} />
                {loopFinalNext?.message ? <div className="rounded border border-slate-100 bg-slate-50 p-2">{String(loopFinalNext.message).slice(0, 260)}</div> : null}
                {loopFinalSelected?.risk ? <div className="rounded border border-red-100 bg-red-50 p-2 text-red-700">{String(loopFinalSelected.risk).slice(0, 260)}</div> : null}
              </div>
              <div className="mt-3 border-t border-slate-100 pt-3">
                <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Reusable Lesson", "可复用经验")}</div>
                <div className="rounded-md border border-emerald-100 bg-emerald-50 p-2 text-xs leading-5 text-emerald-800">
                  {String(loopLesson?.lesson ?? latestStoredLesson?.lesson ?? tx(locale, "No lesson written yet.", "暂无经验记录。")).slice(0, 520)}
                </div>
                <div className="mt-2 text-[11px] text-slate-500">
                  {tx(locale, "Latest stop", "最近停止原因")}: {String(loopLesson?.stop_reason ?? latestStoredLesson?.stop_reason ?? scientistLoop?.stop_reason ?? "not_run")}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Continuation Status", "科学家续跑状态")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Tracks unfinished AI Scientist turns: completed safe tools, remaining safe tools, and the next resumable command.",
                "追踪未完成的 AI Scientist 回合：已完成安全工具、剩余安全工具和下一条可恢复命令。"
              )}
            </CardDescription>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="secondary" onClick={() => void refreshScientistContinuationStatus()} disabled={busy || scientistContinuationBusy}>
              {scientistContinuationBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <RefreshCcw className="h-4 w-4" />}
              {tx(locale, "Refresh Status", "刷新状态")}
            </Button>
            <Button size="sm" variant="secondary" onClick={() => void runScientistContinuationResume()} disabled={busy || scientistContinuationBusy}>
              {scientistContinuationBusy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {tx(locale, "Resume Safe Tools", "自动续跑安全工具")}
            </Button>
            <Button size="sm" variant="primary" onClick={() => void executeScientistNextAction()} disabled={busy || autopilotBusy}>
              {busy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {tx(locale, "Execute Safe Next", "执行安全下一步")}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.8fr_1.2fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Status", "状态")}
              value={<StatusBadge tone={parityPhaseTone(scientistContinuationStatus?.status)}>{scientistContinuationStatus?.status ?? "not_run"}</StatusBadge>}
            />
            <Row label={tx(locale, "Task", "任务")} value={scientistContinuationStatus?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Progress", "进度")} value={`${continuationCompletedCount} / ${continuationTotal}`} />
            <div className="mb-2 h-2 overflow-hidden rounded-full bg-slate-200">
              <div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${Math.max(0, Math.min(100, Math.round(continuationRatio * 100)))}%` }} />
            </div>
            <Row label={tx(locale, "Remaining", "剩余工具")} value={scientistContinuationStatus?.remaining_count ?? continuationRemaining.length} />
            <Row label={tx(locale, "Next Command", "下一条命令")} value={<span className="font-mono text-[11px] text-blue-700">{continuationNextCommand}</span>} />
            <Row label={tx(locale, "Continuation", "续跑证据")} value={scientistContinuationStatus?.continuation_artifact_path ?? ".xsci/scientist_continuation.json"} />
            <Row label={tx(locale, "Status Artifact", "状态证据")} value={scientistContinuationStatus?.artifact_path ?? ".xsci/scientist_continuation_status.json"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistContinuationStatus?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistContinuationStatus?.official_submit ?? "blocked_until_explicit_human_approval"} />
            {scientistContinuationResume ? (
              <>
                <Row label={tx(locale, "Last Resume", "最近自动续跑")} value={<StatusBadge tone={parityPhaseTone(scientistContinuationResume.status)}>{scientistContinuationResume.status ?? "not_run"}</StatusBadge>} />
                <Row label={tx(locale, "Resume Steps", "自动续跑步数")} value={`${scientistContinuationResume.steps_executed ?? 0} / ${scientistContinuationResume.max_steps ?? 0}`} />
                <Row label={tx(locale, "Stop Reason", "停止原因")} value={scientistContinuationResume.stop_reason ?? "(none)"} />
                <Row label={tx(locale, "Resume Artifact", "自动续跑证据")} value={scientistContinuationResume.artifact_path ?? ".xsci/scientist_continuation_resume.json"} />
              </>
            ) : null}
            {scientistContinuationStatus?.message ? (
              <div className="mt-3 rounded-md border border-slate-200 bg-white p-2 text-xs leading-5 text-slate-700">{scientistContinuationStatus.message}</div>
            ) : null}
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Remaining Safe Tools", "剩余安全工具")}</div>
              <div className="max-h-60 space-y-2 overflow-y-auto pr-1">
                {continuationRemaining.length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No remaining safe tools, or no continuation artifact yet.", "暂无剩余安全工具，或尚未生成续跑证据。")}</div>
                ) : (
                  continuationRemaining.map((tool) => (
                    <div key={tool} className="rounded-md border border-amber-100 bg-amber-50 px-3 py-2 font-mono text-[11px] font-semibold text-amber-800">{tool}</div>
                  ))
                )}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Completed Tools", "已完成工具")}</div>
              <div className="max-h-60 space-y-2 overflow-y-auto pr-1">
                {continuationCompleted.length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No completed tool record yet.", "暂无已完成工具记录。")}</div>
                ) : (
                  continuationCompleted.map((tool) => (
                    <div key={tool} className="rounded-md border border-emerald-100 bg-emerald-50 px-3 py-2 font-mono text-[11px] font-semibold text-emerald-800">{tool}</div>
                  ))
                )}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3 md:col-span-2">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-xs font-bold uppercase text-slate-500">{tx(locale, "Auto Resume Trace", "自动续跑轨迹")}</div>
                <StatusBadge tone={parityPhaseTone(scientistContinuationResume?.status)}>{scientistContinuationResume?.status ?? "not_run"}</StatusBadge>
              </div>
              <div className="mb-3 max-h-44 space-y-2 overflow-y-auto pr-1">
                {scientistContinuationResume?.steps?.length ? (
                  scientistContinuationResume.steps.slice(-6).map((item, index) => (
                    <div key={`${item.executed_tool ?? "resume"}-${item.index ?? index}`} className="rounded-md border border-blue-100 bg-blue-50/60 px-3 py-2 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="truncate font-mono font-bold text-blue-800">#{item.index ?? index + 1} {item.executed_tool ?? "(no tool)"}</div>
                          <div className="mt-1 truncate font-mono text-[11px] text-blue-700">{item.selected_command ?? "evomind next"}</div>
                        </div>
                        <StatusBadge tone={item.status?.includes("blocked") ? "amber" : item.status?.includes("executed") ? "green" : "blue"}>{item.status ?? "unknown"}</StatusBadge>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-xs text-slate-400">{tx(locale, "Run Resume Safe Tools to see bounded continuation steps.", "点击自动续跑安全工具后，这里会显示有界续跑步骤。")}</div>
                )}
              </div>
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Continuation Progress History", "续跑进展记录")}</div>
              <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                {continuationHistory.length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "Refresh after a Scientist turn or Safe Next action to see progress records.", "在科学家回合或安全下一步后刷新，可看到进展记录。")}</div>
                ) : (
                  continuationHistory.slice(-8).map((item, index) => (
                    <div key={`${item.safe_tool ?? "tool"}-${item.updated_at ?? index}`} className="rounded-md border border-slate-100 bg-slate-50/70 px-3 py-2 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="truncate font-bold text-slate-800">{item.safe_tool ?? `tool_${index + 1}`}</div>
                          <div className="mt-1 font-mono text-[11px] text-slate-500">{item.updated_at ?? "(no timestamp)"}</div>
                        </div>
                        <StatusBadge tone={item.tool_ok === false ? "red" : item.status?.includes("blocked") ? "amber" : "green"}>{item.status ?? (item.tool_ok === false ? "failed" : "ok")}</StatusBadge>
                      </div>
                      {item.tool_artifact_path ? <div className="mt-2 break-all font-mono text-[11px] text-blue-700">{item.tool_artifact_path}</div> : null}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Action Queue", "科学家行动队列")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Executable next-step queue synthesized from diagnosis, memory, workplan, repair plan, and execution contract.",
                "由诊断、记忆、工作计划、修复计划和执行契约合成的下一步行动队列。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void executeScientistNextAction()} disabled={busy || autopilotBusy}>
            {busy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {tx(locale, "Execute Safe Next", "执行安全下一步")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.8fr_1.2fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Task", "任务")} value={scientistActionQueue?.selected_task || autopilot?.selected_task || selectedTask || "(none)"} />
            <Row label="Trace" value={scientistActionQueue?.trace_run_id ?? autopilot?.trace_run_id ?? "(none)"} />
            <Row label={tx(locale, "Actions", "动作数")} value={scientistActionQueue?.actions?.length ?? autopilot?.action_queue?.length ?? 0} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistActionQueue?.artifact_path ?? autopilot?.action_queue_artifact_path ?? ".xsci/scientist_action_queue.json"} />
            <Row label={tx(locale, "Training", "训练")} value={scientistActionQueue?.no_training_started === false ? "started" : "not_started"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistActionQueue?.official_submit ?? "blocked_until_explicit_human_approval"} />
            <div className="mt-3 rounded-md border border-slate-200 bg-white p-2">
              <Row
                label={tx(locale, "Safe Next", "安全下一步")}
                value={<StatusBadge tone={scientistNextAction?.status === "executed_read_only_tool" ? "green" : scientistNextAction?.status === "blocked_by_gate" ? "amber" : scientistNextAction?.present ? "slate" : "slate"}>{scientistNextAction?.status ?? "not_run"}</StatusBadge>}
              />
              <Row label={tx(locale, "Selected", "选中动作")} value={scientistNextAction?.selected_action?.id ?? "(none)"} />
              <Row label={tx(locale, "Executed Tool", "已执行工具")} value={scientistNextAction?.executed_tool ?? "(none)"} />
              <Row label={tx(locale, "Next Artifact", "下一步证据")} value={scientistNextAction?.artifact_path ?? ".xsci/scientist_next_action.json"} />
              {scientistNextAction?.message ? (
                <div className="mt-2 text-xs leading-5 text-slate-600">{scientistNextAction.message}</div>
              ) : null}
              {scientistNextAction?.blocked_reason ? (
                <div className="mt-2 rounded border border-amber-100 bg-amber-50 p-2 text-xs leading-5 text-amber-800">{scientistNextAction.blocked_reason}</div>
              ) : null}
            </div>
            <div className="mt-3 rounded-md border border-amber-100 bg-amber-50 p-2 text-xs leading-5 text-amber-800">
              {tx(
                locale,
                "Ready commands are still gated: EvoMind shows the next command, but long training and official submit remain policy-controlled.",
                "ready 命令仍受门禁控制：EvoMind 会展示下一步命令，但长训练和官方提交仍由策略门禁控制。"
              )}
            </div>
          </div>
          <div className="max-h-96 space-y-2 overflow-y-auto pr-1">
            {((scientistActionQueue?.actions ?? autopilot?.action_queue) ?? []).length === 0 ? (
              <div className="rounded-md border border-slate-200 bg-white p-3 text-xs text-slate-400">
                {tx(locale, "Run Scientist Autopilot to generate the action queue.", "运行科学家诊断后会生成行动队列。")}
              </div>
            ) : (
              ((scientistActionQueue?.actions ?? autopilot?.action_queue) ?? []).slice(0, 6).map((action, index) => (
                <div key={`${action.id ?? index}`} className="rounded-md border border-slate-200 bg-white p-3 text-xs">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="font-bold text-slate-900">{action.title ?? action.id ?? tx(locale, "Action", "动作")}</div>
                      <div className="mt-1 font-mono text-[11px] font-semibold text-blue-700">{action.command ?? "(no command)"}</div>
                    </div>
                    <StatusBadge tone={actionStatusTone(action.status)}>{action.status ?? "unknown"}</StatusBadge>
                  </div>
                  <div className="mt-2 grid gap-2 md:grid-cols-2">
                    <div className="rounded border border-slate-100 bg-slate-50 p-2">
                      <div className="font-bold uppercase text-slate-500">Gate</div>
                      <div className="mt-1 text-slate-700">{action.gate ?? "none"}</div>
                    </div>
                    <div className="rounded border border-slate-100 bg-slate-50 p-2">
                      <div className="font-bold uppercase text-slate-500">Autonomy</div>
                      <div className="mt-1 text-slate-700">{action.autonomy ?? "guarded"}</div>
                    </div>
                  </div>
                  {action.why ? <div className="mt-2 text-slate-600">{action.why}</div> : null}
                  {action.metadata?.selected_hypothesis && typeof action.metadata.selected_hypothesis === "object" ? (
                    <div className="mt-2 rounded border border-blue-100 bg-blue-50 p-2 text-blue-800">
                      <div className="mb-1 font-bold uppercase text-blue-700">{tx(locale, "Reviewed Hypothesis", "已评审假设")}</div>
                      <div className="grid gap-1 md:grid-cols-2">
                        <div>
                          <span className="font-bold text-blue-700">{tx(locale, "Strategy", "策略")}: </span>
                          {String((action.metadata.selected_hypothesis as Record<string, unknown>).strategy_name ?? (action.metadata.selected_hypothesis as Record<string, unknown>).hypothesis_id ?? "unknown")}
                        </div>
                        <div>
                          <span className="font-bold text-blue-700">Score: </span>
                          {String((action.metadata.selected_hypothesis as Record<string, unknown>).score ?? "n/a")}
                        </div>
                        <div>
                          <span className="font-bold text-blue-700">Branch: </span>
                          {String((action.metadata.selected_hypothesis as Record<string, unknown>).branch_type ?? "n/a")}
                        </div>
                        <div>
                          <span className="font-bold text-blue-700">Mode: </span>
                          {String((action.metadata.selected_hypothesis as Record<string, unknown>).code_generation_mode ?? "n/a")}
                        </div>
                      </div>
                    </div>
                  ) : null}
                  {action.risk ? <div className="mt-2 rounded border border-red-100 bg-red-50 p-2 text-red-700">{action.risk}</div> : null}
                  {action.rollback_condition ? (
                    <div className="mt-2 text-slate-500">
                      <span className="font-bold">{tx(locale, "Rollback", "回滚")}: </span>
                      {action.rollback_condition}
                    </div>
                  ) : null}
                  {(action.expected_artifacts ?? []).length ? (
                    <div className="mt-2 border-t border-slate-100 pt-2">
                      <div className="mb-1 font-bold uppercase text-slate-500">{tx(locale, "Expected Artifacts", "预期证据")}</div>
                      <div className="grid gap-1">
                        {action.expected_artifacts?.slice(0, 5).map((artifact, artifactIndex) => (
                          <div key={`${artifact}-${artifactIndex}`} className="break-all font-mono text-[11px] text-slate-600">- {artifact}</div>
                        ))}
                      </div>
                    </div>
                  ) : null}
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Recovery Snapshot", "科学家恢复快照")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Restartable long-horizon context: guard, turn ledger, step trace, blockers, selected resume action, and exact resume commands.",
                "面向长任务的断点恢复上下文：guard、回合账本、步骤轨迹、阻塞项、选中恢复动作和可直接执行的恢复命令。"
              )}
            </CardDescription>
          </div>
          <Button size="sm" variant="secondary" onClick={() => void runScientistRecovery()} disabled={busy || autopilotBusy}>
            <History className="h-4 w-4" />
            {tx(locale, "Build Snapshot", "生成快照")}
          </Button>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.85fr_1.15fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Decision", "恢复决策")}
              value={
                <StatusBadge tone={scientistRecovery?.recovery_decision === "blocked_clear_gates" ? "amber" : scientistRecovery?.present ? "green" : "slate"}>
                  {scientistRecovery?.recovery_decision ?? "not_run"}
                </StatusBadge>
              }
            />
            <Row label={tx(locale, "Task", "任务")} value={scientistRecovery?.selected_task || selectedTask || "(none)"} />
            <Row label={tx(locale, "Recent Turns", "最近回合")} value={scientistRecovery?.recent_turn_count ?? 0} />
            <Row label={tx(locale, "Recent Steps", "最近步骤")} value={scientistRecovery?.recent_step_count ?? 0} />
            <Row label={tx(locale, "Loop Stop", "循环停止原因")} value={String(scientistRecovery?.latest_loop?.stop_reason ?? "unknown")} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistRecovery?.artifact_path ?? ".xsci/scientist_recovery_snapshot.json"} />
            <Row label={tx(locale, "Guard", "恢复 Guard")} value={scientistRecovery?.guard_path ?? ".xsci/recovery_guard.md"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistRecovery?.official_submit ?? "blocked_until_explicit_human_approval"} />
            <div className="mt-3 rounded-md border border-blue-100 bg-blue-50 p-2 text-xs leading-5 text-blue-800">
              {tx(
                locale,
                "This is a recovery and planning artifact only. It never starts model training and never submits to Kaggle.",
                "这是恢复与规划证据，不会启动模型训练，也不会提交 Kaggle。"
              )}
            </div>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Selected Resume Action", "选中恢复动作")}</div>
              {!scientistRecovery?.selected_resume_action ? (
                <div className="text-xs text-slate-400">{tx(locale, "No ready resume action yet.", "暂无可恢复动作。")}</div>
              ) : (
                <div className="space-y-2 text-xs text-slate-700">
                  <div className="font-bold text-slate-900">{scientistRecovery.selected_resume_action.title ?? scientistRecovery.selected_resume_action.id}</div>
                  <div className="font-mono text-[11px] font-semibold text-blue-700">{scientistRecovery.selected_resume_action.command ?? "(no command)"}</div>
                  <div><span className="font-bold text-slate-500">Gate</span>: {scientistRecovery.selected_resume_action.gate ?? "read_only"}</div>
                  <div><span className="font-bold text-slate-500">Why</span>: {String(scientistRecovery.selected_resume_action.why ?? "").slice(0, 240)}</div>
                </div>
              )}
              {(scientistRecovery?.blockers ?? []).length ? (
                <div className="mt-3 rounded border border-amber-100 bg-amber-50 p-2 text-xs leading-5 text-amber-800">
                  <div className="mb-1 font-bold">{tx(locale, "Blockers", "阻塞项")}</div>
                  {scientistRecovery?.blockers?.slice(0, 8).map((item, index) => <div key={`${item}-${index}`}>- {item}</div>)}
                </div>
              ) : null}
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Resume Commands", "恢复命令")}</div>
              <div className="max-h-36 space-y-1 overflow-y-auto pr-1">
                {(scientistRecovery?.resume_commands ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "Build a recovery snapshot to get commands.", "生成恢复快照后会显示命令。")}</div>
                ) : (
                  scientistRecovery?.resume_commands?.slice(0, 8).map((command, index) => (
                    <div key={`${command}-${index}`} className="rounded border border-slate-100 bg-slate-50 px-2 py-1 font-mono text-[11px] font-semibold text-blue-700">
                      {command}
                    </div>
                  ))
                )}
              </div>
              <div className="mt-3 border-t border-slate-100 pt-3">
                <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Linked Artifacts", "关联证据")}</div>
                <div className="max-h-40 space-y-1 overflow-y-auto pr-1 text-[11px] text-slate-600">
                  {[
                    scientistRecovery?.latest_workplan_artifact,
                    scientistRecovery?.latest_repair_artifact,
                    scientistRecovery?.latest_contract_artifact,
                    scientistRecovery?.action_queue_artifact
                  ].filter(Boolean).map((artifact, index) => (
                    <div key={`${artifact}-${index}`} className="break-all font-mono">- {artifact}</div>
                  ))}
                  {[
                    scientistRecovery?.latest_workplan_artifact,
                    scientistRecovery?.latest_repair_artifact,
                    scientistRecovery?.latest_contract_artifact,
                    scientistRecovery?.action_queue_artifact
                  ].filter(Boolean).length === 0 ? (
                    <div className="text-xs text-slate-400">{tx(locale, "No linked artifacts yet.", "暂无关联证据。")}</div>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{tx(locale, "Scientist Workplan", "科学家工作计划")}</CardTitle>
          <CardDescription>
            {tx(
              locale,
              "Recoverable multi-step plan with gates, evidence, current focus, and resume commands.",
              "可恢复的多步研究计划：步骤、门禁、证据、当前焦点和恢复命令。"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.85fr_1.15fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Mode", "模式")}
              value={<StatusBadge tone={scientistWorkplan?.mode === "ready_for_gated_execution" ? "green" : scientistWorkplan?.present ? "amber" : "slate"}>{scientistWorkplan?.mode ?? "not_run"}</StatusBadge>}
            />
            <Row label={tx(locale, "Autonomy", "自主级别")} value={scientistWorkplan?.autonomy_level ?? "planner_only"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistWorkplan?.artifact_path ?? ".xsci/scientist_workplan.json"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistWorkplan?.official_submit ?? "blocked_until_explicit_human_approval"} />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Current Focus", "当前焦点")}</div>
              <div className="space-y-1 text-xs text-slate-700">
                <div><span className="font-bold text-slate-500">Step</span>: {String(scientistWorkplan?.current_focus?.step_id ?? "not_run")}</div>
                <div><span className="font-bold text-slate-500">Status</span>: {String(scientistWorkplan?.current_focus?.status ?? "not_run")}</div>
                <div><span className="font-bold text-slate-500">Action</span>: {String(scientistWorkplan?.current_focus?.action ?? "").slice(0, 260)}</div>
                {scientistWorkplan?.current_focus?.blocked_reason ? (
                  <div className="rounded border border-red-100 bg-red-50 p-2 text-red-700">
                    {String(scientistWorkplan.current_focus.blocked_reason).slice(0, 260)}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Steps", "步骤")}</div>
              <div className="grid grid-cols-4 gap-2 text-center text-xs">
                {["completed", "ready", "pending", "blocked"].map((key) => (
                  <div key={key} className="rounded border border-slate-100 bg-slate-50 p-2">
                    <div className="font-bold text-slate-900">{String(scientistWorkplan?.summary?.[key] ?? 0)}</div>
                    <div className="text-slate-500">{key}</div>
                  </div>
                ))}
              </div>
              <div className="mt-3 max-h-32 space-y-1 overflow-y-auto pr-1">
                {(scientistWorkplan?.steps ?? []).slice(0, 6).map((step, index) => (
                  <div key={`${String(step.id ?? index)}`} className="flex items-start justify-between gap-2 rounded border border-slate-100 px-2 py-1 text-xs">
                    <span className="min-w-0 truncate font-semibold text-slate-700">{String(step.title ?? step.id ?? "")}</span>
                    <StatusBadge tone={step.status === "completed" ? "green" : step.status === "ready" ? "blue" : step.status === "blocked" ? "red" : "amber"}>{String(step.status ?? "pending")}</StatusBadge>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{tx(locale, "Scientist Repair Plan", "科学家自我修复计划")}</CardTitle>
          <CardDescription>
            {tx(
              locale,
              "Root-cause diagnosis with ordered repair steps, gates, evidence, and a safe next command.",
              "基于真实轨迹和门禁归因阻塞，生成可执行修复步骤、证据要求和安全下一条命令。"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.85fr_1.15fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label={tx(locale, "Mode", "模式")}
              value={<StatusBadge tone={scientistRepairPlan?.mode === "ready_to_execute_guarded" ? "green" : scientistRepairPlan?.mode === "blocked_repair" ? "red" : scientistRepairPlan?.present ? "amber" : "slate"}>{scientistRepairPlan?.mode ?? "not_run"}</StatusBadge>}
            />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistRepairPlan?.artifact_path ?? ".xsci/scientist_repair_plan.json"} />
            <Row label={tx(locale, "Safe Next", "安全下一步")} value={scientistRepairPlan?.safe_next_command ?? "evomind autopilot"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistRepairPlan?.official_submit ?? "blocked_until_explicit_human_approval"} />
            <div className="mt-3 rounded-md border border-blue-100 bg-blue-50 p-2 text-xs text-blue-800">
              {tx(locale, "This plan is read-only. It diagnoses and plans; training still requires a gated run command.", "该计划只读：只诊断和规划；训练仍必须通过门禁 run 命令发起。")}
            </div>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Root Causes", "根因")}</div>
              <div className="max-h-36 space-y-1 overflow-y-auto pr-1 text-xs text-slate-700">
                {(scientistRepairPlan?.root_causes ?? []).length === 0 ? (
                  <div className="text-slate-400">{tx(locale, "Run diagnosis to infer root causes.", "运行诊断后会推断根因。")}</div>
                ) : (
                  scientistRepairPlan?.root_causes?.slice(0, 8).map((item, index) => <div key={`${item}-${index}`}>- {item}</div>)
                )}
              </div>
              <div className="mt-3 border-t border-slate-100 pt-3">
                <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Diagnosis", "诊断")}</div>
                <div className="max-h-32 space-y-1 overflow-y-auto pr-1">
                  {(scientistRepairPlan?.diagnosis ?? []).slice(0, 5).map((issue, index) => (
                    <div key={`${String(issue.root_cause ?? index)}`} className="rounded border border-slate-100 px-2 py-1 text-xs">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-semibold text-slate-700">{String(issue.root_cause ?? "issue")}</span>
                        <StatusBadge tone={issue.severity === "blocker" ? "red" : issue.severity === "warning" ? "amber" : "slate"}>{String(issue.severity ?? "info")}</StatusBadge>
                      </div>
                      <div className="mt-1 text-slate-500">{String(issue.evidence ?? "").slice(0, 180)}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Repair Steps", "修复步骤")}</div>
              <div className="max-h-72 space-y-2 overflow-y-auto pr-1">
                {(scientistRepairPlan?.repair_steps ?? []).length === 0 ? (
                  <div className="text-xs text-slate-400">{tx(locale, "No repair steps yet.", "暂无修复步骤。")}</div>
                ) : (
                  scientistRepairPlan?.repair_steps?.slice(0, 8).map((step, index) => (
                    <div key={`${String(step.id ?? index)}`} className="rounded-md border border-slate-100 bg-slate-50/70 px-3 py-2 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0 font-bold text-slate-800">{String(step.title ?? step.id ?? "")}</div>
                        <StatusBadge tone={step.status === "ready" ? "blue" : step.status === "blocked" ? "red" : step.status === "completed" ? "green" : "amber"}>{String(step.status ?? "pending")}</StatusBadge>
                      </div>
                      <div className="mt-1 text-slate-600">{String(step.action ?? "").slice(0, 220)}</div>
                      {step.command ? <div className="mt-1 font-mono text-[11px] font-semibold text-blue-700">{String(step.command)}</div> : null}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{tx(locale, "Scientist Execution Contract", "科学家执行契约")}</CardTitle>
          <CardDescription>
            {tx(
              locale,
              "Pre-execution go/no-go contract for AgentSession entry, model-training readiness, rollback, and required evidence.",
              "执行前 go/no-go 契约：判断是否可进入 AgentSession、是否真正可训练、回滚条件和必须产出的证据。"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.85fr_1.15fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row
              label="Go / No-Go"
              value={
                <StatusBadge tone={scientistExecutionContract?.go_no_go === "go" ? "green" : scientistExecutionContract?.go_no_go === "no_go" ? "red" : scientistExecutionContract?.go_no_go === "conditional_go_data_contract_first" ? "amber" : "slate"}>
                  {scientistExecutionContract?.go_no_go ?? "not_run"}
                </StatusBadge>
              }
            />
            <Row
              label={tx(locale, "Gate Decision", "执行门禁")}
              value={<StatusBadge tone={executionGateTone}>{executionGateStatus}</StatusBadge>}
            />
            <Row
              label={tx(locale, "AgentSession", "AgentSession")}
              value={<StatusBadge tone={scientistExecutionContract?.agent_session_ready ? "green" : "amber"}>{scientistExecutionContract?.agent_session_ready ? "ready" : "not_ready"}</StatusBadge>}
            />
            <Row
              label={tx(locale, "Model Training", "模型训练")}
              value={<StatusBadge tone={scientistExecutionContract?.model_training_ready ? "green" : "amber"}>{scientistExecutionContract?.model_training_ready ? "ready" : "guarded"}</StatusBadge>}
            />
            <Row label={tx(locale, "Data Contract", "数据契约")} value={scientistExecutionContract?.data_contract_status ?? "unknown"} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistExecutionContract?.artifact_path ?? ".xsci/scientist_execution_contract.json"} />
            <Row label={tx(locale, "Official Submit", "官方提交")} value={scientistExecutionContract?.official_submit ?? "blocked_until_explicit_human_approval"} />
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="text-xs font-bold uppercase text-slate-500">{tx(locale, "Execution Gate", "执行门禁")}</div>
                <StatusBadge tone={executionGateTone}>{executionGateStatus}</StatusBadge>
              </div>
              <div className="space-y-2 text-xs text-slate-700">
                <div>
                  <span className="font-bold text-slate-500">{tx(locale, "Training", "训练状态")}</span>:{" "}
                  {executionGateNoTrainingStarted ? "no_training_started" : "execution_allowed_by_gate"}
                </div>
                <div>
                  <span className="font-bold text-slate-500">{tx(locale, "Submit", "提交")}</span>:{" "}
                  {executionGateOfficialSubmit}
                </div>
                <div>
                  <div className="mb-1 font-bold text-slate-500">{tx(locale, "Blocked By", "阻断来源")}</div>
                  {executionGateBlockedBy.length === 0 ? (
                    <div className="text-slate-400">{tx(locale, "No blockers recorded.", "未记录阻断项。")}</div>
                  ) : (
                    <div className="flex flex-wrap gap-1">
                      {executionGateBlockedBy.slice(0, 6).map((item, index) => (
                        <StatusBadge key={`${item}-${index}`} tone="red">{item}</StatusBadge>
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  <div className="mb-1 font-bold text-slate-500">{tx(locale, "Safe Next", "安全下一步")}</div>
                  {executionGateSafeCommands.length === 0 ? (
                    <div className="font-mono text-[11px] text-blue-700">evomind contract</div>
                  ) : (
                    <div className="space-y-1">
                      {executionGateSafeCommands.slice(0, 4).map((command, index) => (
                        <div key={`${command}-${index}`} className="rounded border border-blue-100 bg-blue-50 px-2 py-1 font-mono text-[11px] font-semibold text-blue-700">
                          {command}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                {executionGateDecision?.message ? (
                  <div className="rounded border border-slate-100 bg-slate-50 p-2 text-slate-600">
                    {executionGateDecision.message.slice(0, 260)}
                  </div>
                ) : null}
                {executionGateSetupBlockers.length ? (
                  <div className="rounded border border-amber-100 bg-amber-50 p-2 text-amber-800">
                    <div className="font-bold">{tx(locale, "Setup Blockers", "配置阻断")}</div>
                    {executionGateSetupBlockers.slice(0, 2).map((item, index) => <div key={`${item}-${index}`}>- {item.slice(0, 220)}</div>)}
                  </div>
                ) : null}
              </div>
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Decision", "决策")}</div>
              <div className="space-y-1 text-xs text-slate-700">
                <div><span className="font-bold text-slate-500">Action</span>: {String(scientistExecutionContract?.decision?.selected_action ?? "not_run")}</div>
                <div><span className="font-bold text-slate-500">Branch</span>: {String(scientistExecutionContract?.decision?.selected_branch ?? "not_run")}</div>
                <div><span className="font-bold text-slate-500">Code Mode</span>: {String(scientistExecutionContract?.decision?.code_generation_mode ?? "not_run")}</div>
                <div><span className="font-bold text-slate-500">Rollback</span>: {String(scientistExecutionContract?.rollback_condition ?? "hold if gates fail").slice(0, 220)}</div>
                <div><span className="font-bold text-slate-500">Command</span>: <span className="font-mono text-[11px] text-blue-700">{scientistExecutionContract?.execution_command ?? "evomind contract"}</span></div>
              </div>
              {executionGateRootCauses.length ? (
                <div className="mt-3 rounded border border-amber-100 bg-amber-50 p-2 text-xs text-amber-800">
                  <div className="font-bold">{tx(locale, "Root Causes", "根因")}</div>
                  {executionGateRootCauses.slice(0, 5).map((item, index) => <div key={`${item}-${index}`}>- {item}</div>)}
                </div>
              ) : null}
            </div>
            <div className="rounded-md border border-slate-200 bg-white p-3">
              <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Required Evidence", "必须证据")}</div>
              <div className="max-h-44 space-y-1 overflow-y-auto pr-1 text-xs text-slate-700">
                {(scientistExecutionContract?.required_artifacts ?? []).length === 0 ? (
                  <div className="text-slate-400">{tx(locale, "Run diagnosis to build the contract.", "运行诊断以生成执行契约。")}</div>
                ) : (
                  scientistExecutionContract?.required_artifacts?.slice(0, 10).map((item, index) => <div key={`${item}-${index}`}>- {item}</div>)
                )}
              </div>
              <div className="mt-3 rounded-md border border-blue-100 bg-blue-50 p-2 text-xs text-blue-800">
                {scientistExecutionContract?.claim_boundary ?? tx(locale, "No leaderboard, rank, medal, or top30 claim is allowed without Kaggle response evidence.", "没有 Kaggle response 证据时，不允许声明排行榜、排名、奖牌或 top30。")}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Live Scientist Stream", "科学家实时执行流")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Auto-refreshing trace of tool calls, gates, artifacts, and recovery decisions. This panel is read-only and never starts training.",
                "自动刷新工具调用、门禁、证据文件和恢复决策；此面板只读，不会启动训练。"
              )}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <StatusBadge tone={streamTransportTone}>{streamTransportLabel}</StatusBadge>
            <StatusBadge tone={streamTone}>{streamStatus}</StatusBadge>
            <Button size="sm" variant="secondary" onClick={() => void refreshScientistStream()}>
              <RefreshCcw className="h-3.5 w-3.5" />
              {tx(locale, "Refresh", "刷新")}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="grid gap-3 xl:grid-cols-[0.72fr_1.28fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Events", "事件数")} value={scientistStream?.event_count ?? 0} />
            <Row label={tx(locale, "Latest Tool", "最新工具")} value={latestStreamEvent?.tool || "(none)"} />
            <Row label={tx(locale, "Latest Phase", "最新阶段")} value={latestStreamEvent?.phase || "not_run"} />
            <Row label={tx(locale, "Latest Gate", "最新门禁")} value={latestStreamEvent?.gate || "(none)"} />
            <Row label={tx(locale, "Last Refresh", "最后刷新")} value={streamLastUpdated} />
            <Row label={tx(locale, "Artifact", "证据文件")} value={scientistStream?.artifact_path ?? ".xsci/scientist_step_trace.jsonl"} />
            <div className="mt-3 rounded-md border border-blue-100 bg-blue-50 p-2 text-xs leading-5 text-blue-800">
              {tx(
                locale,
                "Official submit and leaderboard claims remain blocked unless a human approval and Kaggle response artifact exist.",
                "没有人工批准和 Kaggle response 证据时，官方提交、排名和奖牌声明保持阻断。"
              )}
            </div>
          </div>
          <div className="thin-scrollbar max-h-80 overflow-y-auto rounded-md border border-slate-200 bg-white p-3">
            {streamEvents.length === 0 ? (
              <div className="text-xs text-slate-400">
                {tx(locale, "No Scientist stream event yet. Run Autopilot, Workplan, Contract, or a gated run to create trace evidence.", "暂无科学家事件流。运行 Autopilot、Workplan、Contract 或受控 run 后会生成轨迹证据。")}
              </div>
            ) : (
              <div className="space-y-2">
                {streamEvents.slice(-14).reverse().map((event, index) => {
                  const status = String(event.status ?? "info");
                  const tone: StatusTone = status === "completed" || status === "passed" || status === "ok"
                    ? "green"
                    : status === "blocked" || status === "failed"
                      ? "red"
                      : status === "running"
                        ? "amber"
                        : "blue";
                  return (
                    <div key={`${event.event_id ?? index}`} className="rounded-md border border-slate-100 bg-slate-50/80 px-3 py-2 text-xs">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="min-w-0 font-bold text-slate-800">
                          {event.phase || "step"} {event.tool ? <span className="font-semibold text-slate-500">/ {event.tool}</span> : null}
                        </div>
                        <StatusBadge tone={tone}>{status}</StatusBadge>
                      </div>
                      <div className="mt-1 text-slate-600">{String(event.message ?? "").slice(0, 300)}</div>
                      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] font-semibold text-slate-500">
                        {event.ts ? <span>{event.ts}</span> : null}
                        {event.gate ? <span>gate={event.gate}</span> : null}
                        {event.artifact_path ? <span className="max-w-full truncate text-blue-700">artifact={event.artifact_path}</span> : null}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{tx(locale, "Scientist Turn Ledger", "科学家回合账本")}</CardTitle>
          <CardDescription>
            {tx(
              locale,
              "Recoverable trace of recent research-agent turns, tools, decisions, and evidence.",
              "可恢复的研究 Agent 回合轨迹：工具、决策、证据和回答摘要。"
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.85fr_1.15fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Artifact", "账本文件")} value={scientistTurns?.artifact_path ?? ".xsci/scientist_turns.jsonl"} />
            <Row label={tx(locale, "Recent Turns", "最近回合")} value={scientistTurns?.count ?? 0} />
            <Row label={tx(locale, "Route", "路线")} value={String(scientistTurns?.latest?.route ?? "not_run")} />
            <Row label={tx(locale, "Task", "任务")} value={String(scientistTurns?.latest?.task ?? selectedTask ?? "(none)")} />
          </div>
          <div className="rounded-md border border-slate-200 bg-white p-3">
            <div className="mb-2 text-xs font-bold uppercase text-slate-500">{tx(locale, "Latest Turn", "最近一次回合")}</div>
            {!scientistTurns?.latest ? (
              <div className="text-xs text-slate-400">{tx(locale, "No scientist turn has been recorded yet.", "还没有记录科学家回合。")}</div>
            ) : (
              <div className="space-y-2 text-xs text-slate-700">
                <div><span className="font-bold text-slate-500">User</span>: {String(scientistTurns.latest.user ?? "").slice(0, 180)}</div>
                <div><span className="font-bold text-slate-500">Tools</span>: {Array.isArray(scientistTurns.latest.forced_tools) ? scientistTurns.latest.forced_tools.join(", ") : "(none)"}</div>
                <div><span className="font-bold text-slate-500">Preview</span>: {String(scientistTurns.latest.answer_preview ?? "").slice(0, 360)}</div>
                <div><span className="font-bold text-slate-500">Submit</span>: {String(scientistTurns.latest.official_submit ?? "blocked_until_explicit_human_approval")}</div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle>{tx(locale, "Scientist Step Trace", "科学家步骤轨迹")}</CardTitle>
            <CardDescription>
              {tx(
                locale,
                "Low-level event stream for each tool call, gate, artifact, and workplan step.",
                "展示每一次工具调用、门禁判断、证据文件和工作计划步骤的低层事件流。"
              )}
            </CardDescription>
          </div>
          <StatusBadge tone={scientistStepTrace?.present ? "green" : "slate"}>
            {scientistStepTrace?.present ? "recording" : "not_run"}
          </StatusBadge>
        </CardHeader>
        <CardContent className="grid gap-3 lg:grid-cols-[0.75fr_1.25fr]">
          <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <Row label={tx(locale, "Trace File", "轨迹文件")} value={scientistStepTrace?.artifact_path ?? ".xsci/scientist_step_trace.jsonl"} />
            <Row label={tx(locale, "Recent Events", "最近事件")} value={scientistStepTrace?.count ?? 0} />
            <Row label={tx(locale, "Latest Phase", "最新阶段")} value={String(scientistStepTrace?.latest?.phase ?? "not_run")} />
            <Row label={tx(locale, "Latest Status", "最新状态")} value={String(scientistStepTrace?.latest?.status ?? "not_run")} />
          </div>
          <div className="thin-scrollbar max-h-72 overflow-y-auto rounded-md border border-slate-200 bg-white p-3">
            {(scientistStepTrace?.recent ?? []).length === 0 ? (
              <div className="text-xs text-slate-400">
                {tx(locale, "Run Scientist Autopilot to create the first step trace.", "运行科学家诊断后会生成第一条步骤轨迹。")}
              </div>
            ) : (
              <div className="space-y-2">
                {(scientistStepTrace?.recent ?? []).slice(-12).reverse().map((event, index) => {
                  const status = String(event.status ?? "info");
                  const tone: StatusTone = status === "completed" || status === "ok" ? "green" : status === "blocked" || status === "failed" ? "red" : "blue";
                  return (
                    <div key={`${String(event.event_id ?? index)}`} className="rounded-md border border-slate-100 bg-slate-50/70 px-3 py-2 text-xs">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="min-w-0 font-bold text-slate-800">
                          {String(event.phase ?? "step")} {event.tool ? <span className="font-semibold text-slate-500">/ {String(event.tool)}</span> : null}
                        </div>
                        <StatusBadge tone={tone}>{status}</StatusBadge>
                      </div>
                      <div className="mt-1 text-slate-600">{String(event.message ?? "").slice(0, 260)}</div>
                      {event.artifact_path ? (
                        <div className="mt-1 truncate text-[11px] font-semibold text-blue-700">{String(event.artifact_path)}</div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>{tx(locale, "Command Input", "命令输入")}</CardTitle>
              <CardDescription>{tx(locale, "Type a command or use a quick action.", "输入自然语言命令，或使用下方快捷动作。")}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <textarea
                className="w-full rounded-md border border-slate-200 bg-white p-3 text-sm text-slate-800 shadow-sm focus:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-200"
                rows={3}
                placeholder={tx(locale, "e.g. Create a workstation run for playground_series_s6e6", "例如：为 playground_series_s6e6 创建工作站 run")}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    parseInput();
                  }
                }}
              />
              <Button onClick={() => parseInput()} disabled={!input.trim()}>
                <Send className="h-4 w-4" />
                {tx(locale, "Parse Command", "解析命令")}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{tx(locale, "Quick Actions", "快捷动作")}</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {quickActions.map((item) => (
                <Button key={item.label} size="sm" variant="secondary" onClick={() => quick(item.label, item.command)} disabled={busy}>
                  <item.icon className="h-3.5 w-3.5" />
                  {item.label}
                </Button>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{tx(locale, "Page Shortcuts", "页面跳转")}</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {[
                ["gates", tx(locale, "Open Gates", "进入 Gate")],
                ["code", tx(locale, "Open Code Studio", "进入代码工作台")],
                ["report", tx(locale, "Open Report Studio", "进入报告工作台")]
              ].map(([page, label]) => (
                <Button key={page} size="sm" variant="ghost" onClick={() => navigateTo(page)}>
                  {label}
                </Button>
              ))}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <div ref={previewRef}>
            <Card>
              <CardHeader>
                <CardTitle>{tx(locale, "Command Preview", "命令预览")}</CardTitle>
                <CardDescription>{tx(locale, "Review intent and risk before execution.", "执行前确认意图、风险和后端动作。")}</CardDescription>
              </CardHeader>
              <CardContent>
                {!parsed ? (
                  <p className="text-xs text-slate-400">{tx(locale, "No command parsed yet.", "尚未解析命令。")}</p>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <StatusBadge tone={riskTone(parsed.risk)}>{parsed.risk}</StatusBadge>
                      <span className="text-xs font-bold text-slate-700">{parsed.intent}</span>
                    </div>
                    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                      <Row label={tx(locale, "Intent", "意图")} value={parsed.intent} />
                      <Row label={tx(locale, "Task ID", "任务 ID")} value={parsed.taskId} />
                      <Row label={tx(locale, "Risk", "风险")} value={parsed.risk} />
                      <Row label={tx(locale, "Description", "说明")} value={parsed.description} />
                      {parsed.blockedReason && <Row label={tx(locale, "Blocked", "阻断原因")} value={<span className="text-red-600">{parsed.blockedReason}</span>} />}
                    </div>
                    {parsed.risk !== "blocked" ? (
                      <Button variant={parsed.risk === "gated" ? "secondary" : "primary"} onClick={executeAction} disabled={busy}>
                        {busy ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                        {parsed.risk === "gated" ? tx(locale, "Submit to Gate", "提交到 Gate") : tx(locale, "Execute", "执行")}
                      </Button>
                    ) : (
                      <Button disabled variant="danger">
                        <XCircle className="h-4 w-4" />
                        {tx(locale, "Blocked", "已阻断")}
                      </Button>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {lastResult && (
            <Card>
              <CardHeader>
                <CardTitle>{tx(locale, "Execution Result", "执行结果")}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                  <Row label={tx(locale, "Action", "动作")} value={lastResult.action} />
                  <Row label={tx(locale, "Status", "状态")} value={lastResult.ok ? <span className="text-emerald-600">OK</span> : <span className="text-red-600">{lastResult.error ?? "failed"}</span>} />
                  {lastResult.artifact && <Row label={tx(locale, "Artifact", "产物")} value={<span>{lastResult.artifact}</span>} />}
                  {lastResult.sessionId && <Row label="Session / Run / Gate ID" value={lastResult.sessionId} />}
                  {!!lastResult.rawResponse && (
                    <div className="mt-3">
                      <div className="mb-2 text-xs font-semibold text-slate-500">Raw JSON Response</div>
                      <JsonInspector data={lastResult.rawResponse} />
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>{tx(locale, "Message History", "消息记录")}</CardTitle>
            </CardHeader>
            <CardContent>
              {messages.length === 0 ? (
                <p className="text-xs text-slate-400">{tx(locale, "No messages yet.", "暂无消息。")}</p>
              ) : (
                <div className="max-h-48 space-y-2 overflow-y-auto">
                  {messages.map((msg, i) => (
                    <div
                      key={`${msg.timestamp}-${i}`}
                      className={`rounded-md px-3 py-2 text-xs ${
                        msg.role === "user" ? "bg-blue-50 text-blue-800" : msg.role === "error" ? "bg-red-50 text-red-700" : "bg-emerald-50 text-emerald-800"
                      }`}
                    >
                      <span className="mr-2 font-bold uppercase">{msg.role}</span>
                      {msg.content}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {lastActionTrace && (
            <Card>
              <CardHeader>
                <CardTitle>{tx(locale, "Latest Action Trace", "最新 Action Trace")}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
                  <Row label={tx(locale, "Action", "动作")} value={lastActionTrace.action} />
                  <Row label={tx(locale, "Message", "消息")} value={lastActionTrace.message} />
                  {lastActionTrace.artifact && <Row label={tx(locale, "Artifact", "产物")} value={<span>{lastActionTrace.artifact}</span>} />}
                  <Row label={tx(locale, "At", "时间")} value={lastActionTrace.at} />
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </main>
  );
}
