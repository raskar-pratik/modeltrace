/**
 * Reusable minimalist-ui components:
 * - Original ModelTrace branching-M SVG mark & wordmark
 * - Semantic Verdict, ExecutionState, DEMO, and ScoreScope badges
 * - Real-data SVG charts (Empirical CDF of z_i vs Uniform(0,1) null, and Per-Prompt Score Shift plot)
 * - Epistemic & Confounder Disclosure Banner
 */

import React from 'react';
import { AuditVerdict, ExecutionState, PromptScore, ScoreMode } from '../types/api';

export function BranchingMIcon(props: { size?: number }) {
  const sz = props.size || 32;
  return (
    <svg
      width={sz}
      height={sz}
      viewBox="0 0 64 64"
      fill="none"
      aria-hidden="true"
      style={{ flexShrink: 0 }}
    >
      <rect
        x="2"
        y="2"
        width="60"
        height="60"
        rx="10"
        fill="#F7F6F3"
        stroke="#111111"
        stroke-width="2.5"
      />
      <path
        d="M14 48V18L26 34L38 18V48"
        stroke="#111111"
        stroke-width="4"
        stroke-linecap="square"
        stroke-linejoin="miter"
      />
      <path
        d="M38 28L50 18V48"
        stroke="#1F6C9F"
        stroke-width="3.5"
        stroke-linecap="square"
        stroke-dasharray="3 2"
      />
      <circle cx="50" cy="18" r="4" fill="#9F2F2D" />
      <circle cx="38" cy="18" r="3.5" fill="#111111" />
    </svg>
  );
}

export function VerdictBadge(props: { verdict: AuditVerdict | string }) {
  const v = props.verdict;
  if (v === 'INCONSISTENT') {
    return <span className="badge badge-red">VERDICT: INCONSISTENT</span>;
  }
  if (v === 'CONSISTENT_WITHIN_TESTED_LIMITS') {
    return (
      <span className="badge badge-sage">VERDICT: CONSISTENT WITHIN TESTED LIMITS</span>
    );
  }
  return <span className="badge badge-apricot">VERDICT: INCONCLUSIVE</span>;
}

export function ExecutionStateBadge(props: { state: ExecutionState | string }) {
  const s = props.state;
  if (s === 'COMPLETED') {
    return <span className="badge badge-sage">STATE: COMPLETED</span>;
  }
  if (s === 'RUNNING' || s === 'CREATED') {
    return <span className="badge badge-blue">STATE: {s}</span>;
  }
  if (s === 'FAILED') {
    return <span className="badge badge-red">STATE: FAILED</span>;
  }
  return <span className="badge badge-apricot">STATE: {s}</span>;
}

export function DemoBadge(props: { isDemo?: boolean }) {
  if (!props.isDemo) return null;
  return <span className="badge badge-blue">DEMO SYNTHETIC</span>;
}

export function ScoreScopeBadge(props: { scoreMode: ScoreMode | string }) {
  if (props.scoreMode === 'length_distribution_only') {
    return (
      <span className="badge badge-apricot">SCOPE: LENGTH DISTRIBUTION ONLY</span>
    );
  }
  return <span className="badge badge-neutral">SCOPE: LOCAL LOGIT RANK</span>;
}

export function EpistemicNoticeBar() {
  return (
    <div
      className="bento-card bento-card-muted"
      style={{ padding: '14px 18px', marginBottom: '20px', fontSize: '13px' }}
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
        <span className="mono" style={{ fontWeight: 600, color: '#111111' }}>
          EPISTEMIC BOUNDARY:
        </span>
        <span style={{ color: '#2F3437' }}>
          Audits test observable output behavior under a frozen configuration—not proof of model weights, parameter count, precision, or provider intent. Chat templates, hidden system prompts, moderation filters, or GPU kernel reductions can trigger INCONSISTENT verdicts even when base weights match.
        </span>
      </div>
    </div>
  );
}

/**
 * Real-data SVG Chart 1: Empirical CDF of finite-sample uniformized ranks z_i vs Uniform(0,1) diagonal.
 */
export function EmpiricalCdfChart(props: {
  promptScores: PromptScore[];
  cvmStatistic?: number | null;
  pValue?: number | null;
}) {
  const scores = props.promptScores || [];
  if (scores.length === 0) {
    return (
      <div
        style={{
          padding: '28px',
          textAlign: 'center',
          color: '#787774',
          border: '1px dashed #EAEAEA',
          borderRadius: '6px',
        }}
      >
        No stored prompt rank coordinates (z_i) available for this audit.
      </div>
    );
  }

  const sortedZ = scores
    .map((s) => Math.max(0, Math.min(1, Number(s.z_uniform))))
    .sort((a, b) => a - b);
  const n = sortedZ.length;

  const width = 460;
  const height = 250;
  const padL = 44;
  const padR = 20;
  const padT = 18;
  const padB = 34;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const toX = (z: number) => padL + z * plotW;
  const toY = (cdf: number) => padT + (1 - cdf) * plotH;

  const stepPoints: string[] = [`${toX(0).toFixed(1)},${toY(0).toFixed(1)}`];
  for (let i = 0; i < n; i++) {
    const zVal = sortedZ[i];
    const prevF = i / n;
    const nextF = (i + 1) / n;
    stepPoints.push(`${toX(zVal).toFixed(1)},${toY(prevF).toFixed(1)}`);
    stepPoints.push(`${toX(zVal).toFixed(1)},${toY(nextF).toFixed(1)}`);
  }
  stepPoints.push(`${toX(1).toFixed(1)},${toY(1).toFixed(1)}`);

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%', height: 'auto', display: 'block' }}
        role="img"
        aria-label="Empirical cumulative distribution of uniformized prompt ranks z_i compared against Uniform(0,1) null diagonal"
      >
        <rect
          x={padL}
          y={padT}
          width={plotW}
          height={plotH}
          fill="#F9F9F8"
          stroke="#EAEAEA"
          stroke-width="1"
        />

        {[0.25, 0.5, 0.75].map((tick) => (
          <g key={tick}>
            <line
              x1={padL}
              y1={toY(tick)}
              x2={padL + plotW}
              y2={toY(tick)}
              stroke="#EAEAEA"
              stroke-dasharray="2 2"
            />
            <line
              x1={toX(tick)}
              y1={padT}
              x2={toX(tick)}
              y2={padT + plotH}
              stroke="#EAEAEA"
              stroke-dasharray="2 2"
            />
          </g>
        ))}

        <line
          x1={toX(0)}
          y1={toY(0)}
          x2={toX(1)}
          y2={toY(1)}
          stroke="#787774"
          stroke-width="1.5"
          stroke-dasharray="4 3"
        />

        <polyline
          fill="none"
          stroke="#111111"
          stroke-width="2"
          points={stepPoints.join(' ')}
        />

        {sortedZ.map((zVal, idx) => {
          const expected = (2 * (idx + 1) - 1) / (2 * n);
          const diff = Math.abs(zVal - expected);
          const dotColor = diff > 0.22 ? '#9F2F2D' : '#1F6C9F';
          return (
            <circle
              key={idx}
              cx={toX(zVal)}
              cy={toY((idx + 0.5) / n)}
              r="3.2"
              fill={dotColor}
            />
          );
        })}

        <text x={padL - 8} y={toY(0) + 4} text-anchor="end" font-size="9.5" fill="#787774" className="mono">
          0.0
        </text>
        <text x={padL - 8} y={toY(0.5) + 4} text-anchor="end" font-size="9.5" fill="#787774" className="mono">
          0.5
        </text>
        <text x={padL - 8} y={toY(1.0) + 4} text-anchor="end" font-size="9.5" fill="#787774" className="mono">
          1.0
        </text>

        <text x={toX(0)} y={height - 10} text-anchor="start" font-size="9.5" fill="#787774" className="mono">
          0.0
        </text>
        <text x={toX(0.5)} y={height - 10} text-anchor="middle" font-size="9.5" fill="#787774" className="mono">
          Uniformized Rank z_i (Null = Diagonal)
        </text>
        <text x={toX(1)} y={height - 10} text-anchor="end" font-size="9.5" fill="#787774" className="mono">
          1.0
        </text>
      </svg>
      <div
        className="mono"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '11.5px',
          color: '#787774',
          marginTop: '6px',
        }}
      >
        <span>n = {n} prompts</span>
        <span>
          CvM W² = {props.cvmStatistic != null ? props.cvmStatistic.toFixed(4) : '—'}
        </span>
        <span>
          MC p = {props.pValue != null ? props.pValue.toFixed(5) : '—'}
        </span>
      </div>
    </div>
  );
}

/**
 * Real-data SVG Chart 2: Per-Prompt Score Shift (Target Score - Reference Mean Score)
 */
export function PromptScoreShiftChart(props: { promptScores: PromptScore[] }) {
  const scores = props.promptScores || [];
  if (scores.length === 0) {
    return (
      <div
        style={{
          padding: '28px',
          textAlign: 'center',
          color: '#787774',
          border: '1px dashed #EAEAEA',
          borderRadius: '6px',
        }}
      >
        No per-prompt score shifts recorded.
      </div>
    );
  }

  const maxAbs = Math.max(
    0.5,
    ...scores.map((s) => Math.abs(Number(s.score_shift) || 0))
  );

  const width = 460;
  const height = 250;
  const padL = 48;
  const padR = 18;
  const padT = 18;
  const padB = 34;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const zeroY = padT + plotH / 2;

  const toY = (shift: number) => {
    const norm = Math.max(-1, Math.min(1, shift / maxAbs));
    return zeroY - norm * (plotH / 2 - 10);
  };

  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%', height: 'auto', display: 'block' }}
        role="img"
        aria-label="Per-prompt score shift chart showing target minus mean reference score across prompts"
      >
        <rect
          x={padL}
          y={padT}
          width={plotW}
          height={plotH}
          fill="#F9F9F8"
          stroke="#EAEAEA"
          stroke-width="1"
        />
        <line
          x1={padL}
          y1={zeroY}
          x2={padL + plotW}
          y2={zeroY}
          stroke="#111111"
          stroke-width="1"
          stroke-dasharray="3 2"
        />

        {scores.map((s, idx) => {
          const x =
            padL +
            (scores.length === 1
              ? plotW / 2
              : (idx / (scores.length - 1)) * (plotW - 24) + 12);
          const y = toY(Number(s.score_shift));
          const isHigh = Math.abs(Number(s.z_uniform) - 0.5) > 0.35;
          const color = isHigh ? '#9F2F2D' : '#1F6C9F';
          return (
            <g key={s.prompt_id}>
              <line
                x1={x}
                y1={zeroY}
                x2={x}
                y2={y}
                stroke={color}
                stroke-width="2"
              />
              <circle cx={x} cy={y} r="3.5" fill={color} />
            </g>
          );
        })}

        <text x={padL - 6} y={padT + 10} text-anchor="end" font-size="9.5" fill="#787774" className="mono">
          +{maxAbs.toFixed(2)}
        </text>
        <text x={padL - 6} y={zeroY + 4} text-anchor="end" font-size="9.5" fill="#111111" className="mono">
          0.00
        </text>
        <text x={padL - 6} y={padT + plotH - 4} text-anchor="end" font-size="9.5" fill="#787774" className="mono">
          -{maxAbs.toFixed(2)}
        </text>
        <text x={padL + plotW / 2} y={height - 10} text-anchor="middle" font-size="9.5" fill="#787774" className="mono">
          Prompts (1..{scores.length}) &middot; Target − Mean Reference Shift
        </text>
      </svg>
      <div
        className="mono"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: '11.5px',
          color: '#787774',
          marginTop: '6px',
        }}
      >
        <span>Mode: {scores[0]?.score_mode}</span>
        <span>
          Mean Shift:{' '}
          {(
            scores.reduce((acc, s) => acc + Number(s.score_shift), 0) /
            Math.max(1, scores.length)
          ).toFixed(4)}
        </span>
      </div>
    </div>
  );
}
