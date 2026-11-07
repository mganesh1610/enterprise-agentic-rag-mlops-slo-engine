import React, { useState } from "react";

export interface SLOMetrics {
  serviceName: string;
  availabilityPct: number;
  sloAvailabilityTargetPct: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  sloP95TargetMs: number;
  psiDrift: number;
  ksDrift: number;
  canaryStatus: "PROMOTED_HEALTHY" | "ROLLED_BACK_TO_CHAMPION";
}

export const OperationalReadinessConsole: React.FC<{ metrics: SLOMetrics }> = ({ metrics }) => {
  const [query, setQuery] = useState("Triage high-touchpoint network routing incident with SLA breach risk");

  const isHealthy =
    metrics.availabilityPct >= metrics.sloAvailabilityTargetPct &&
    metrics.p95LatencyMs <= metrics.sloP95TargetMs &&
    metrics.psiDrift < 0.2;

  return (
    <div style={{ fontFamily: "sans-serif", padding: "20px", maxWidth: "960px", margin: "0 auto" }}>
      <h2>Enterprise AI Operational Readiness, Hybrid RAG &amp; SLO Console</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px", marginBottom: "18px" }}>
        <div style={{ border: "1px solid #d0d7de", borderRadius: "6px", padding: "12px" }}>
          <strong>Availability SLO</strong>
          <div>{metrics.availabilityPct.toFixed(2)}% (Target: {metrics.sloAvailabilityTargetPct}%)</div>
        </div>
        <div style={{ border: "1px solid #d0d7de", borderRadius: "6px", padding: "12px" }}>
          <strong>Latency (p50 / p95)</strong>
          <div>{metrics.p50LatencyMs} ms / {metrics.p95LatencyMs} ms</div>
        </div>
        <div style={{ border: "1px solid #d0d7de", borderRadius: "6px", padding: "12px" }}>
          <strong>Model Drift (PSI / KS)</strong>
          <div>PSI: {metrics.psiDrift.toFixed(4)} | KS: {metrics.ksDrift.toFixed(4)}</div>
        </div>
        <div style={{ border: "1px solid #d0d7de", borderRadius: "6px", padding: "12px" }}>
          <strong>Canary &amp; Rollback Gate</strong>
          <div style={{ color: isHealthy ? "#1a7f37" : "#cf222e", fontWeight: 700 }}>
            {metrics.canaryStatus}
          </div>
        </div>
      </div>
      <div>
        <label htmlFor="agent-query"><strong>Agentic RAG Triage Query:</strong></label>
        <input
          id="agent-query"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ width: "100%", padding: "8px", marginTop: "6px" }}
        />
      </div>
    </div>
  );
};
