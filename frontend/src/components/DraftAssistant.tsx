import { useState, useMemo } from 'react'
import {
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend,
} from 'recharts'
import type { PredictionRow } from '../api'

/* ── colours ─────────────────────────────────────────────────── */
const POS_COLOR: Record<string, string> = {
  QB: '#00f5ff', RB: '#a78bfa', WR: '#10b981', TE: '#fbbf24',
}

/* ── snake-draft pick calculator ─────────────────────────────── */
function snakePicks(draftPos: number, numTeams: number, rounds: number): number[] {
  const picks: number[] = []
  for (let r = 1; r <= rounds; r++) {
    if (r % 2 === 1) picks.push((r - 1) * numTeams + draftPos)      // odd → forward
    else picks.push(r * numTeams + 1 - draftPos)                     // even → reverse
  }
  return picks
}

/* ── replacement-level ranks per position ────────────────────── */
function replacementRanks(numTeams: number) {
  return {
    QB: numTeams + 1,            // 1 starter per team
    RB: Math.round(numTeams * 2.5), // 2 starters + flex share
    WR: Math.round(numTeams * 2.5), // 2 starters + flex share
    TE: numTeams + 1,            // 1 starter per team
  }
}

/* ── types ───────────────────────────────────────────────────── */
interface DraftPlayer {
  name: string
  position: string
  team: string
  projection: number        // 18-week projection (native scale)
  vor: number               // value over replacement (same scale)
  vorNorm: number           // 0-1 normalised VOR for cross-position ranking
  posRank: number
  overallRank: number
  likelyRound: number       // estimated draft round (1-indexed)
  isMyPick: boolean         // likely available at one of user's picks
  matchup: string
}

/* ── main component ──────────────────────────────────────────── */
interface DraftAssistantProps {
  allData: Record<string, PredictionRow[]>
  weekLabel: string
}

export function DraftAssistant({ allData }: DraftAssistantProps) {
  const [draftPos, setDraftPos] = useState(6)
  const [numTeams, setNumTeams] = useState(12)
  const [numRounds, setNumRounds] = useState(15)
  const [activeSection, setActiveSection] = useState<'board' | 'picks' | 'scarcity'>('board')

  /* snake picks for the user */
  const myPicks = useMemo(() => snakePicks(draftPos, numTeams, numRounds), [draftPos, numTeams, numRounds])

  /* compute VOR and draft board */
  const { board, byPosition, posScarcity } = useMemo(() => {
    const repRanks = replacementRanks(numTeams)
    const totalPicks = numTeams * numRounds
    const posMap: Record<string, { name: string; position: string; team: string; projection: number; matchup: string }[]> = {}

    // gather and sort by projection within each position
    for (const pos of ['QB', 'RB', 'WR', 'TE']) {
      const rows = allData[pos] ?? []
      const sorted = rows
        .map((r) => {
          const proj = r.projection_18w != null ? Number(r.projection_18w)
            : r.projected_points != null ? Number(r.projected_points) : null
          if (proj == null || Number.isNaN(proj)) return null
          const opp = r.upcoming_opponent ?? ''
          const ha = r.upcoming_home_away ?? ''
          let matchup = '—'
          if (opp) matchup = ha === 'home' ? `vs ${opp}` : ha === 'away' ? `@ ${opp}` : opp
          return {
            name: String(r.name ?? 'Unknown'),
            position: pos,
            team: String(r.team ?? '—'),
            projection: proj,
            matchup,
          }
        })
        .filter(Boolean) as { name: string; position: string; team: string; projection: number; matchup: string }[]
      sorted.sort((a, b) => b.projection - a.projection)
      posMap[pos] = sorted
    }

    // compute VOR within each position
    const byPosition: Record<string, DraftPlayer[]> = {}
    const allPlayers: DraftPlayer[] = []

    for (const pos of ['QB', 'RB', 'WR', 'TE']) {
      const sorted = posMap[pos] ?? []
      const repIdx = Math.min((repRanks[pos as keyof typeof repRanks] ?? 13) - 1, sorted.length - 1)
      const repValue = repIdx >= 0 && sorted[repIdx] ? sorted[repIdx].projection : 0

      const posPlayers: DraftPlayer[] = sorted.map((p, i) => ({
        ...p,
        vor: p.projection - repValue,
        vorNorm: 0,  // will be filled after
        posRank: i + 1,
        overallRank: 0,
        likelyRound: 0,
        isMyPick: false,
        matchup: p.matchup,
      }))
      byPosition[pos] = posPlayers
      allPlayers.push(...posPlayers)
    }

    // normalise VOR across positions for cross-position ranking
    // use min-max within each position, then compare
    for (const pos of ['QB', 'RB', 'WR', 'TE']) {
      const posP = byPosition[pos] ?? []
      const maxVor = posP.length > 0 ? Math.max(...posP.map((p) => p.vor)) : 1
      const minVor = posP.length > 0 ? Math.min(...posP.map((p) => p.vor)) : 0
      const range = maxVor - minVor || 1
      posP.forEach((p) => {
        p.vorNorm = (p.vor - minVor) / range
      })
    }

    // rank by normalised VOR across all positions
    allPlayers.sort((a, b) => b.vorNorm - a.vorNorm)
    allPlayers.forEach((p, i) => {
      p.overallRank = i + 1
      p.likelyRound = Math.ceil(p.overallRank / numTeams)
      // mark if player is likely available at one of user's picks
      p.isMyPick = myPicks.some((pick) => {
        const pickRound = Math.ceil(pick / numTeams)
        return p.likelyRound === pickRound
      })
    })

    // trim to draftable pool
    const board = allPlayers.slice(0, totalPicks + 20)

    // positional scarcity data: VOR by rank within position
    const posScarcity: { rank: number; QB: number; RB: number; WR: number; TE: number }[] = []
    const maxLen = Math.max(
      ...(Object.values(byPosition).map((arr) => Math.min(arr.length, 40)))
    )
    for (let i = 0; i < maxLen; i++) {
      posScarcity.push({
        rank: i + 1,
        QB: byPosition.QB?.[i]?.vor ?? 0,
        RB: byPosition.RB?.[i]?.vor ?? 0,
        WR: byPosition.WR?.[i]?.vor ?? 0,
        TE: byPosition.TE?.[i]?.vor ?? 0,
      })
    }

    return { board, byPosition, posScarcity }
  }, [allData, numTeams, numRounds, myPicks])

  /* round-by-round recommendations */
  const roundRecs = useMemo(() => {
    return myPicks.map((overallPick, idx) => {
      const round = idx + 1
      // players likely available at this pick: overall rank near this pick number
      const available = board.filter((p) => {
        // a player is "likely available" if their overall rank is within a reasonable window
        return p.overallRank >= overallPick - Math.floor(numTeams * 0.3)
          && p.overallRank <= overallPick + Math.floor(numTeams * 0.5)
      })
      // sort by VOR descending to recommend the best
      available.sort((a, b) => b.vorNorm - a.vorNorm)
      // also get "steal" candidates: players ranked higher than pick but might fall
      const steals = board.filter((p) =>
        p.overallRank < overallPick && p.overallRank >= overallPick - numTeams
      ).sort((a, b) => b.vorNorm - a.vorNorm).slice(0, 3)
      return {
        round,
        overallPick,
        recommended: available.slice(0, 5),
        steals,
      }
    })
  }, [myPicks, board, numTeams])

  const hasData = board.length > 0

  return (
    <div className="draft-assistant">
      {/* Settings */}
      <div className="section-card draft-settings">
        <h2 className="section-heading">Draft Assistant</h2>
        <p style={{ color: 'var(--color-text-muted)', marginBottom: '1rem' }}>
          Configure your league settings. The draft board uses <strong>Value Over Replacement (VOR)</strong> based
          on our ML model's 18-week projections to rank players across all positions and recommend optimal picks
          for your draft slot.
        </p>
        <div className="draft-settings__row">
          <div className="draft-settings__group">
            <label className="draft-settings__label">Your Draft Position</label>
            <div className="draft-settings__input-row">
              <input
                type="range"
                min={1}
                max={numTeams}
                value={draftPos}
                onChange={(e) => setDraftPos(Number(e.target.value))}
                className="draft-settings__slider"
              />
              <span className="draft-settings__value">{draftPos} of {numTeams}</span>
            </div>
          </div>
          <div className="draft-settings__group">
            <label className="draft-settings__label">Number of Teams</label>
            <select
              value={numTeams}
              onChange={(e) => {
                const n = Number(e.target.value)
                setNumTeams(n)
                if (draftPos > n) setDraftPos(n)
              }}
              className="draft-settings__select"
            >
              {[8, 10, 12, 14, 16].map((n) => (
                <option key={n} value={n}>{n} teams</option>
              ))}
            </select>
          </div>
          <div className="draft-settings__group">
            <label className="draft-settings__label">Rounds</label>
            <select
              value={numRounds}
              onChange={(e) => setNumRounds(Number(e.target.value))}
              className="draft-settings__select"
            >
              {[13, 14, 15, 16, 17, 18].map((n) => (
                <option key={n} value={n}>{n} rounds</option>
              ))}
            </select>
          </div>
        </div>

        {/* Snake pick preview */}
        <div className="draft-picks-preview">
          <span className="draft-picks-preview__label">Your picks:</span>
          {myPicks.map((pick, i) => (
            <span key={i} className="draft-picks-preview__pick">
              Rd {i + 1} <span className="draft-picks-preview__num">#{pick}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Section tabs */}
      <div className="draft-tabs">
        <button
          className={`draft-tabs__btn ${activeSection === 'board' ? 'draft-tabs__btn--active' : ''}`}
          onClick={() => setActiveSection('board')}
        >
          Draft Board
        </button>
        <button
          className={`draft-tabs__btn ${activeSection === 'picks' ? 'draft-tabs__btn--active' : ''}`}
          onClick={() => setActiveSection('picks')}
        >
          My Round-by-Round Picks
        </button>
        <button
          className={`draft-tabs__btn ${activeSection === 'scarcity' ? 'draft-tabs__btn--active' : ''}`}
          onClick={() => setActiveSection('scarcity')}
        >
          Positional Scarcity
        </button>
      </div>

      {!hasData && (
        <div className="section-card" style={{ textAlign: 'center', padding: '3rem' }}>
          <p style={{ color: 'var(--color-text-muted)' }}>
            No projection data available. Run <code>python scripts/generate_app_data.py</code> then refresh.
          </p>
        </div>
      )}

      {/* ── DRAFT BOARD ──────────────────────────────────────── */}
      {hasData && activeSection === 'board' && (
        <div className="section-card draft-board">
          <h3 className="section-heading">
            VOR Draft Board
            <span className="section-heading__sub">
              {board.length} players · {numTeams}-team league · Pick #{draftPos}
            </span>
          </h3>
          <p style={{ color: 'var(--color-text-muted)', fontSize: 'var(--text-small)', marginBottom: '1rem' }}>
            Players ranked by cross-position Value Over Replacement. Highlighted rows align with your estimated draft picks.
            QB scored by fantasy points; RB/WR/TE by utilization score. VOR normalised for cross-position comparison.
          </p>
          <div className="draft-board__wrap">
            <table className="draft-board__table">
              <thead>
                <tr>
                  <th className="draft-board__th">#</th>
                  <th className="draft-board__th">Est. Round</th>
                  <th className="draft-board__th">Player</th>
                  <th className="draft-board__th">Pos</th>
                  <th className="draft-board__th">Team</th>
                  <th className="draft-board__th draft-board__th--right">Projection</th>
                  <th className="draft-board__th draft-board__th--right">VOR</th>
                  <th className="draft-board__th draft-board__th--right">Pos Rank</th>
                </tr>
              </thead>
              <tbody>
                {board.slice(0, numTeams * numRounds).map((p) => {
                  const isHighlight = myPicks.some((pick) => Math.abs(pick - p.overallRank) <= 2)
                  return (
                    <tr
                      key={`${p.name}-${p.position}`}
                      className={`draft-board__tr ${isHighlight ? 'draft-board__tr--highlight' : ''}`}
                    >
                      <td className="draft-board__td draft-board__td--rank">{p.overallRank}</td>
                      <td className="draft-board__td">
                        <span className="draft-board__round">Rd {p.likelyRound}</span>
                      </td>
                      <td className="draft-board__td draft-board__td--name">{p.name}</td>
                      <td className="draft-board__td">
                        <span className="draft-board__pos" style={{ color: POS_COLOR[p.position] ?? '#94a3b8' }}>
                          {p.position}
                        </span>
                      </td>
                      <td className="draft-board__td">{p.team}</td>
                      <td className="draft-board__td draft-board__td--right">
                        <span className="draft-board__proj">{p.projection.toFixed(1)}</span>
                      </td>
                      <td className="draft-board__td draft-board__td--right">
                        <span style={{ color: p.vor > 0 ? '#10b981' : '#94a3b8' }}>
                          {p.vor > 0 ? '+' : ''}{p.vor.toFixed(1)}
                        </span>
                      </td>
                      <td className="draft-board__td draft-board__td--right">
                        {p.position}{p.posRank}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <div className="legend-row" style={{ marginTop: '0.75rem' }}>
            {['QB', 'RB', 'WR', 'TE'].map((pos) => (
              <span key={pos} className="legend-item">
                <span className="legend-dot" style={{ background: POS_COLOR[pos] }} />
                {pos}
              </span>
            ))}
            <span className="legend-item" style={{ marginLeft: '1.5rem' }}>
              <span style={{ display: 'inline-block', width: 10, height: 10, background: 'rgba(0,245,255,0.08)', border: '1px solid rgba(0,245,255,0.25)', borderRadius: 2, marginRight: 6 }} />
              Near your pick
            </span>
          </div>
        </div>
      )}

      {/* ── ROUND-BY-ROUND PICKS ────────────────────────────── */}
      {hasData && activeSection === 'picks' && (
        <div className="draft-rounds">
          <h3 className="section-heading" style={{ marginBottom: '1rem' }}>
            Round-by-Round Targets
            <span className="section-heading__sub">Pick #{draftPos} in {numTeams}-team league</span>
          </h3>
          <p style={{ color: 'var(--color-text-muted)', fontSize: 'var(--text-small)', marginBottom: '1.5rem' }}>
            For each of your draft picks, we show the highest-VOR players likely to be available.
            <strong> "If available" steals</strong> are players ranked above your pick who might slip — draft them immediately if they're on the board.
          </p>
          <div className="draft-rounds__grid">
            {roundRecs.map((rec) => (
              <div key={rec.round} className="section-card draft-round-card">
                <div className="draft-round-card__header">
                  <span className="draft-round-card__round">Round {rec.round}</span>
                  <span className="draft-round-card__pick">Overall Pick #{rec.overallPick}</span>
                </div>

                {rec.steals.length > 0 && (
                  <div className="draft-round-card__steals">
                    <div className="draft-round-card__steals-label">If available — draft immediately:</div>
                    {rec.steals.map((p, j) => (
                      <div key={j} className="draft-round-card__steal-row">
                        <span className="draft-board__pos" style={{ color: POS_COLOR[p.position] }}>{p.position}</span>
                        <span className="draft-round-card__player-name">{p.name}</span>
                        <span className="draft-round-card__player-team">{p.team}</span>
                        <span className="draft-round-card__player-vor" style={{ color: '#10b981' }}>
                          +{p.vor.toFixed(1)} VOR
                        </span>
                      </div>
                    ))}
                  </div>
                )}

                <div className="draft-round-card__recs-label">Recommended targets:</div>
                {rec.recommended.length === 0 ? (
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 'var(--text-small)' }}>
                    No strong targets — best available at this point
                  </div>
                ) : (
                  rec.recommended.map((p, j) => (
                    <div key={j} className="draft-round-card__rec-row">
                      <span className="draft-round-card__rec-num">#{j + 1}</span>
                      <span className="draft-board__pos" style={{ color: POS_COLOR[p.position] }}>{p.position}</span>
                      <span className="draft-round-card__player-name">{p.name}</span>
                      <span className="draft-round-card__player-team">{p.team}</span>
                      <span className="draft-round-card__player-proj">{p.projection.toFixed(1)}</span>
                      <span className="draft-round-card__player-rank">{p.position}{p.posRank}</span>
                    </div>
                  ))
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── POSITIONAL SCARCITY ──────────────────────────────── */}
      {hasData && activeSection === 'scarcity' && (
        <div className="section-card draft-scarcity">
          <h3 className="section-heading">Positional Scarcity (VOR Drop-Off)</h3>
          <p style={{ color: 'var(--color-text-muted)', fontSize: 'var(--text-small)', marginBottom: '1rem' }}>
            How quickly does value decline at each position? <strong>Steep drop-off = draft early</strong> (e.g. RB/TE).
            Flat curves (e.g. WR) mean you can wait and still find comparable value later.
            This analysis is key to deciding which position to prioritize at each pick.
          </p>
          <div style={{ height: 380 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={posScarcity} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="rank"
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8' }}
                  label={{ value: 'Position Rank', position: 'bottom', fill: '#94a3b8', fontSize: 12, offset: 0 }}
                />
                <YAxis
                  stroke="#94a3b8"
                  tick={{ fill: '#94a3b8' }}
                  label={{ value: 'Value Over Replacement', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{ background: '#1a1f3a', border: '1px solid #334155', borderRadius: 8, color: '#cbd5e1' }}
                  formatter={(value: number, name: string) => [value.toFixed(1), name]}
                />
                <Legend />
                <Line type="monotone" dataKey="QB" stroke={POS_COLOR.QB} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="RB" stroke={POS_COLOR.RB} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="WR" stroke={POS_COLOR.WR} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="TE" stroke={POS_COLOR.TE} strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Scarcity summary cards */}
          <div className="draft-scarcity__summary">
            {(['QB', 'RB', 'WR', 'TE'] as const).map((pos) => {
              const posPlayers = byPosition[pos] ?? []
              const top5Avg = posPlayers.slice(0, 5).reduce((s, p) => s + p.vor, 0) / Math.min(5, posPlayers.length || 1)
              const next10Avg = posPlayers.slice(5, 15).reduce((s, p) => s + p.vor, 0) / Math.min(10, Math.max(1, posPlayers.slice(5, 15).length))
              const dropoff = top5Avg - next10Avg
              const repRanks2 = replacementRanks(numTeams)
              const repRank = repRanks2[pos]
              return (
                <div key={pos} className="draft-scarcity__card" style={{ borderColor: POS_COLOR[pos] }}>
                  <div className="draft-scarcity__card-pos" style={{ color: POS_COLOR[pos] }}>{pos}</div>
                  <div className="draft-scarcity__card-stat">
                    <span className="draft-scarcity__card-val">{top5Avg.toFixed(1)}</span>
                    <span className="draft-scarcity__card-label">Top-5 avg VOR</span>
                  </div>
                  <div className="draft-scarcity__card-stat">
                    <span className="draft-scarcity__card-val">{dropoff.toFixed(1)}</span>
                    <span className="draft-scarcity__card-label">Drop-off (5→15)</span>
                  </div>
                  <div className="draft-scarcity__card-stat">
                    <span className="draft-scarcity__card-val">{repRank}</span>
                    <span className="draft-scarcity__card-label">Replacement rank</span>
                  </div>
                  <div className="draft-scarcity__card-advice" style={{ color: POS_COLOR[pos] }}>
                    {dropoff > top5Avg * 0.4 ? 'High scarcity — draft early' : 'Deep position — can wait'}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Strategy note */}
          <div className="draft-scarcity__note">
            <strong>Draft strategy tip:</strong> Positions with steep VOR drop-offs (high scarcity)
            should be prioritized in early rounds. Deep positions with gradual decline can be
            addressed in middle-to-late rounds without significant value loss. At pick #{draftPos},
            consider securing high-scarcity positions when the top tier is still available at your pick.
          </div>
        </div>
      )}
    </div>
  )
}
