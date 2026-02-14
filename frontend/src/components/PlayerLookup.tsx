import { useState, useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts'
import type { PredictionRow } from '../api'

const POSITION_COLORS: Record<string, string> = {
  QB: '#00f5ff',
  RB: '#a78bfa',
  WR: '#10b981',
  TE: '#fbbf24',
  K: '#f472b6',
  DST: '#fb923c',
}

function getProjectedPoints(r: PredictionRow): number | null {
  const v = r.projected_points ?? r.predicted_points ?? r.projection_1w
  return v != null && !Number.isNaN(Number(v)) ? Number(v) : null
}

function getTier(rank: number, total: number): { label: string; color: string; bg: string } {
  const pct = rank / total
  if (pct <= 0.08) return { label: 'Must Start', color: '#10b981', bg: 'rgba(16,185,129,0.15)' }
  if (pct <= 0.25) return { label: 'Start', color: '#00f5ff', bg: 'rgba(0,245,255,0.1)' }
  if (pct <= 0.5) return { label: 'Flex', color: '#fbbf24', bg: 'rgba(251,191,36,0.1)' }
  return { label: 'Sit', color: '#94a3b8', bg: 'rgba(148,163,184,0.08)' }
}

interface PlayerLookupProps {
  allData: Record<string, PredictionRow[]>
  weekLabel: string
}

interface EnrichedPlayer {
  row: PredictionRow
  pts1w: number | null
  pts4w: number | null
  pts18w: number | null
  std: number | null
  util: number | null
  rank: number
  total: number
  tier: ReturnType<typeof getTier>
  posAvg: number
}

export function PlayerLookup({ allData }: PlayerLookupProps) {
  const [searchA, setSearchA] = useState('')
  const [searchB, setSearchB] = useState('')
  const [selectedA, setSelectedA] = useState<string | null>(null)
  const [selectedB, setSelectedB] = useState<string | null>(null)

  const allPlayers = useMemo(() => {
    const map = new Map<string, EnrichedPlayer>()
    for (const pos of ['QB', 'RB', 'WR', 'TE', 'K', 'DST']) {
      const rows = allData[pos] ?? []
      const sorted = rows
        .map((r) => ({ r, pts: getProjectedPoints(r) }))
        .filter((x) => x.pts != null)
        .sort((a, b) => b.pts! - a.pts!)

      const total = sorted.length
      const avg = total > 0 ? sorted.reduce((a, x) => a + x.pts!, 0) / total : 0

      sorted.forEach((x, i) => {
        const name = String(x.r.name ?? '').trim()
        if (!name) return
        map.set(name, {
          row: x.r,
          pts1w: x.r.projection_1w != null ? Number(x.r.projection_1w) : (x.r.projected_points != null ? Number(x.r.projected_points) : null),
          pts4w: x.r.projection_4w != null ? Number(x.r.projection_4w) : null,
          pts18w: x.r.projection_18w != null ? Number(x.r.projection_18w) : null,
          std: x.r.prediction_std != null ? Number(x.r.prediction_std) : (x.r.weekly_volatility != null ? Number(x.r.weekly_volatility) : null),
          util: x.r.utilization_score != null ? Number(x.r.utilization_score) : null,
          rank: i + 1,
          total,
          tier: getTier(i + 1, total),
          posAvg: avg,
        })
      })
    }
    return map
  }, [allData])

  const nameList = useMemo(() => Array.from(allPlayers.keys()).sort(), [allPlayers])

  const filteredA = useMemo(() => {
    if (!searchA.trim()) return []
    const q = searchA.trim().toLowerCase()
    return nameList.filter((n) => n.toLowerCase().includes(q)).slice(0, 8)
  }, [nameList, searchA])

  const filteredB = useMemo(() => {
    if (!searchB.trim()) return []
    const q = searchB.trim().toLowerCase()
    return nameList.filter((n) => n.toLowerCase().includes(q)).slice(0, 8)
  }, [nameList, searchB])

  const playerA = selectedA ? allPlayers.get(selectedA) ?? null : null
  const playerB = selectedB ? allPlayers.get(selectedB) ?? null : null

  const comparisonData = useMemo(() => {
    if (!playerA || !playerB) return null
    const metrics = [
      { key: '1W Proj', a: playerA.pts1w, b: playerB.pts1w },
      { key: '4W Proj', a: playerA.pts4w, b: playerB.pts4w },
      { key: 'ROS Proj', a: playerA.pts18w, b: playerB.pts18w },
    ].filter((m) => m.a != null || m.b != null)
    return metrics
  }, [playerA, playerB])

  const radarData = useMemo(() => {
    if (!playerA || !playerB) return null
    const maxPts = Math.max(playerA.pts1w ?? 0, playerB.pts1w ?? 0, 1)
    const max4w = Math.max(playerA.pts4w ?? 0, playerB.pts4w ?? 0, 1)
    const maxRos = Math.max(playerA.pts18w ?? 0, playerB.pts18w ?? 0, 1)
    const maxUtil = Math.max(playerA.util ?? 0, playerB.util ?? 0, 1)

    return [
      { metric: '1W Pts', A: ((playerA.pts1w ?? 0) / maxPts) * 100, B: ((playerB.pts1w ?? 0) / maxPts) * 100 },
      { metric: '4W Pts', A: ((playerA.pts4w ?? 0) / max4w) * 100, B: ((playerB.pts4w ?? 0) / max4w) * 100 },
      { metric: 'ROS', A: ((playerA.pts18w ?? 0) / maxRos) * 100, B: ((playerB.pts18w ?? 0) / maxRos) * 100 },
      { metric: 'Utilization', A: ((playerA.util ?? 0) / maxUtil) * 100, B: ((playerB.util ?? 0) / maxUtil) * 100 },
      { metric: 'Consistency', A: playerA.std != null ? Math.max(0, 100 - playerA.std * 5) : 50, B: playerB.std != null ? Math.max(0, 100 - playerB.std * 5) : 50 },
    ]
  }, [playerA, playerB])

  function renderPlayerCard(player: EnrichedPlayer | null) {
    if (!player) return null
    const r = player.row
    const opp = r.upcoming_opponent ?? ''
    const ha = r.upcoming_home_away ?? ''
    let matchup = '—'
    if (opp) matchup = ha === 'home' ? `vs ${opp}` : ha === 'away' ? `@ ${opp}` : opp

    return (
      <div className="player-card">
        <div className="player-card__header">
          <div className="player-card__pos-badge" style={{ background: POSITION_COLORS[String(r.position)] ?? '#94a3b8' }}>
            {r.position}
          </div>
          <div>
            <div className="player-card__name">{r.name}</div>
            <div className="player-card__team">{r.team ?? '—'} · {matchup}</div>
          </div>
          <span className="tier-badge" style={{ color: player.tier.color, background: player.tier.bg, marginLeft: 'auto' }}>
            {player.tier.label}
          </span>
        </div>
        <div className="player-card__stats">
          <div className="player-card__stat">
            <div className="player-card__stat-label">1W Projected</div>
            <div className="player-card__stat-value" style={{ color: '#00f5ff' }}>
              {player.pts1w?.toFixed(1) ?? '—'}
            </div>
          </div>
          <div className="player-card__stat">
            <div className="player-card__stat-label">4W Projected</div>
            <div className="player-card__stat-value" style={{ color: '#a78bfa' }}>
              {player.pts4w?.toFixed(1) ?? '—'}
            </div>
          </div>
          <div className="player-card__stat">
            <div className="player-card__stat-label">ROS Projected</div>
            <div className="player-card__stat-value" style={{ color: '#10b981' }}>
              {player.pts18w?.toFixed(1) ?? '—'}
            </div>
          </div>
          {player.util != null && (
            <div className="player-card__stat">
              <div className="player-card__stat-label">Utilization</div>
              <div className="player-card__stat-value" style={{ color: '#fbbf24' }}>
                {player.util.toFixed(1)}
              </div>
            </div>
          )}
        </div>
        <div className="player-card__meta">
          <span>Rank: #{player.rank} of {player.total} ({r.position})</span>
          {player.std != null && <span>Uncertainty: ±{player.std.toFixed(1)} pts</span>}
          <span>Pos avg: {player.posAvg.toFixed(1)} pts</span>
          {player.pts1w != null && (
            <span style={{ color: player.pts1w > player.posAvg ? '#10b981' : '#fb7185' }}>
              {player.pts1w > player.posAvg ? '▲' : '▼'} {Math.abs(player.pts1w - player.posAvg).toFixed(1)} vs avg
            </span>
          )}
        </div>
        {player.pts1w != null && player.std != null && (
          <div className="player-card__range-bar">
            <div className="player-card__range-label">
              <span>Floor: {Math.max(0, player.pts1w - 1.5 * player.std).toFixed(1)}</span>
              <span>Ceiling: {(player.pts1w + 1.5 * player.std).toFixed(1)}</span>
            </div>
            <div className="player-card__range-track">
              <div
                className="player-card__range-fill"
                style={{
                  left: `${Math.max(0, ((player.pts1w - 1.5 * player.std) / (player.pts1w + 1.5 * player.std)) * 100)}%`,
                  width: `${100 - Math.max(0, ((player.pts1w - 1.5 * player.std) / (player.pts1w + 1.5 * player.std)) * 100)}%`,
                }}
              />
              <div
                className="player-card__range-marker"
                style={{ left: `${(player.pts1w / (player.pts1w + 1.5 * player.std)) * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="player-lookup">
      <div className="section-card">
        <h2 className="section-heading">Player Lookup & Compare</h2>
        <p style={{ color: 'var(--color-text-muted)', marginBottom: '1.5rem' }}>
          Search for any player to view detailed projections, or compare two players side-by-side for start/sit decisions.
        </p>

        <div className="player-lookup__search-row">
          <div className="player-lookup__search-col">
            <label className="player-lookup__label">Player A</label>
            <div className="player-lookup__autocomplete">
              <input
                type="text"
                placeholder="Search player..."
                value={searchA}
                onChange={(e) => { setSearchA(e.target.value); setSelectedA(null) }}
                className="player-lookup__input"
              />
              {filteredA.length > 0 && !selectedA && (
                <div className="player-lookup__dropdown">
                  {filteredA.map((n) => {
                    const p = allPlayers.get(n)
                    return (
                      <button key={n} className="player-lookup__option" onClick={() => { setSelectedA(n); setSearchA(n) }}>
                        <span>{n}</span>
                        <span style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>{p?.row.position} · {p?.row.team}</span>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
          <div className="player-lookup__vs">VS</div>
          <div className="player-lookup__search-col">
            <label className="player-lookup__label">Player B (optional)</label>
            <div className="player-lookup__autocomplete">
              <input
                type="text"
                placeholder="Search player..."
                value={searchB}
                onChange={(e) => { setSearchB(e.target.value); setSelectedB(null) }}
                className="player-lookup__input"
              />
              {filteredB.length > 0 && !selectedB && (
                <div className="player-lookup__dropdown">
                  {filteredB.map((n) => {
                    const p = allPlayers.get(n)
                    return (
                      <button key={n} className="player-lookup__option" onClick={() => { setSelectedB(n); setSearchB(n) }}>
                        <span>{n}</span>
                        <span style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>{p?.row.position} · {p?.row.team}</span>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Player Cards */}
      {(playerA || playerB) && (
        <div className="player-lookup__cards">
          {playerA && renderPlayerCard(playerA)}
          {playerB && renderPlayerCard(playerB)}
        </div>
      )}

      {/* Comparison Chart */}
      {playerA && playerB && comparisonData && comparisonData.length > 0 && (
        <div className="section-card">
          <h3 className="section-heading">Head-to-Head Comparison</h3>
          <div className="player-lookup__compare-grid">
            <div style={{ height: 280 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={comparisonData.map((m) => ({
                    metric: m.key,
                    [selectedA!]: m.a ?? 0,
                    [selectedB!]: m.b ?? 0,
                  }))}
                  margin={{ top: 10, right: 20, left: 10, bottom: 20 }}
                >
                  <XAxis dataKey="metric" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
                  <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ background: '#1a1f3a', border: '1px solid #334155', borderRadius: 8, color: '#cbd5e1' }} />
                  <Bar dataKey={selectedA!} fill="#00f5ff" radius={[4, 4, 0, 0]} />
                  <Bar dataKey={selectedB!} fill="#a78bfa" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            {radarData && (
              <div style={{ height: 280 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                    <PolarRadiusAxis tick={false} axisLine={false} />
                    <Radar name={selectedA!} dataKey="A" stroke="#00f5ff" fill="#00f5ff" fillOpacity={0.2} />
                    <Radar name={selectedB!} dataKey="B" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.2} />
                    <Tooltip contentStyle={{ background: '#1a1f3a', border: '1px solid #334155', borderRadius: 8, color: '#cbd5e1' }} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
          <div className="player-lookup__verdict">
            {playerA.pts1w != null && playerB.pts1w != null && (
              <div className="player-lookup__verdict-text">
                <strong style={{ color: playerA.pts1w >= playerB.pts1w ? '#00f5ff' : '#a78bfa' }}>
                  {playerA.pts1w >= playerB.pts1w ? selectedA : selectedB}
                </strong>{' '}
                is projected {Math.abs((playerA.pts1w ?? 0) - (playerB.pts1w ?? 0)).toFixed(1)} pts higher this week
              </div>
            )}
          </div>
        </div>
      )}

      {/* Single player detail when no comparison */}
      {playerA && !playerB && (
        <div className="section-card">
          <h3 className="section-heading">Projection Breakdown</h3>
          <div style={{ height: 200 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={[
                  { horizon: '1 Week', pts: playerA.pts1w ?? 0 },
                  { horizon: '4 Weeks', pts: playerA.pts4w ?? 0 },
                  { horizon: 'ROS', pts: playerA.pts18w ?? 0 },
                ].filter((d) => d.pts > 0)}
                margin={{ top: 10, right: 20, left: 10, bottom: 20 }}
              >
                <XAxis dataKey="horizon" stroke="#94a3b8" tick={{ fill: '#cbd5e1' }} />
                <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8' }} />
                <Tooltip contentStyle={{ background: '#1a1f3a', border: '1px solid #334155', borderRadius: 8, color: '#cbd5e1' }} />
                <Bar dataKey="pts" radius={[4, 4, 0, 0]}>
                  <Cell fill="#00f5ff" />
                  <Cell fill="#a78bfa" />
                  <Cell fill="#10b981" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}
