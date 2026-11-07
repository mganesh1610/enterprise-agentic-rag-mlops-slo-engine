# Schema Catalog, SLO Definitions & Incident Rollback Runbook

## 1. Verified Public Datasets & Provenance

| Dataset | Source URL | Scale & Granularity | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **UCI Incident Management Process Enriched Event Log** | https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log | `141,712` event rows consolidated into `24,918` unique enterprise incidents across `71` assignment groups and `53` categories | `6294e29a311647306bfdfc85783f7df66517c197b9cd49aa5ee36ba9c525d1d6` |
| **PolyAI Banking77 Service Intent Corpus** | https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data | `13,083` regulated customer service utterances (`10,003` train / `3,080` test) across `77` intents | Train: `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` / Test: `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |
| **Backup Public Dataset: Mendeley Helpdesk Event Log** | https://data.mendeley.com/datasets/39bp3vv62t/1 | `13,710` service ticket events across `3,804` operational support cases | Verified backup source |

---

## 2. Warehouse Table Schema (`incidents`)

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `incident_id` | `TEXT PRIMARY KEY` | Unique enterprise incident identifier (`INC...`) |
| `category` | `TEXT` | Operational service category (`53` distinct categories) |
| `assignment_group` | `TEXT` | Responsible engineering/support group (`71` distinct groups) |
| `priority_num` | `INTEGER` | Incident severity tier (`1 = Critical` to `4 = Low`) |
| `reassignment_count` | `INTEGER` | Cumulative ticket handoffs across support groups |
| `reopen_count` | `INTEGER` | Number of times the incident was reopened after resolution |
| `sys_mod_count` | `INTEGER` | Total system state modifications across incident lifecycle |
| `knowledge` | `INTEGER` | Binary flag (`1/0`) indicating knowledge-base article utilization |
| `sla_breach` | `INTEGER` | Target label (`1 = missed SLA`, `0 = met SLA`) |

---

## 3. Service Level Objectives (SLOs) & Automated Canary Rollback Policy

| Metric | Production SLO Target | Canary / Shadow Alert Threshold | Automated Action |
| :--- | :--- | :--- | :--- |
| **Service Availability** | $\ge 99.95\%$ | $< 99.95\%$ or Burn Rate $> 2.0\times$ | Immediate traffic shift to Champion |
| **p50 Inference Latency** | $\le 85.0\text{ ms}$ | $> 100.0\text{ ms}$ over 5-min window | Warning alert to MLOps on-call |
| **p95 Inference Latency** | $\le 150.0\text{ ms}$ | $> 150.0\text{ ms}$ over 5-min window | Automated canary rollback |
| **Prediction Drift (PSI)** | $< 0.10$ | $\text{PSI} \ge 0.20$ | Automated rollback + retraining trigger |
| **Feature/Score KS Drift** | $< 0.08$ | $D_{\text{KS}} \ge 0.12$ | Automated rollback + retraining trigger |
