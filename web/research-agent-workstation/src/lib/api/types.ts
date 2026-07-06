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
