'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { JuliRobot, RobotState } from './JuliRobot';

const TELEMETRY_URL = process.env.NEXT_PUBLIC_TELEMETRY_URL || 'http://127.0.0.1:8080';

interface TelemetryData {
  health: any;
  journal: any;
  brain: any;
  risk: any;
}

function fmt$(v: number) {
  if (v >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(1)}K`;
  return `$${v.toFixed(2)}`;
}

export function JuliDashboard() {
  const [data, setData] = useState<TelemetryData | null>(null);
  const [robotState, setRobotState] = useState<RobotState>({
    emotion: 'idle',
    gaze: 'center',
    recentEvent: null,
    pnl: 0,
  });
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchTelemetry = useCallback(async () => {
    try {
      const [health, journal, brain, risk] = await Promise.all([
        fetch(`${TELEMETRY_URL}/health`).then(r => r.json()).catch(() => null),
        fetch(`${TELEMETRY_URL}/journal`).then(r => r.json()).catch(() => null),
        fetch(`${TELEMETRY_URL}/brain`).then(r => r.json()).catch(() => null),
        fetch(`${TELEMETRY_URL}/risk`).then(r => r.json()).catch(() => null),
      ]);
      setData({ health, journal, brain, risk });
      setError(null);

      const journalData = journal?.journal;
      const openPositions = journalData?.positions?.length || health?.open_positions || 0;
      const todayPnl = journalData?.today_pnl || 0;
      const riskData = risk?.risk;
      const safetyNet = riskData?.safety_enabled ?? true;
      const dailyLoss = riskData?.daily_loss || 0;

      let event: RobotState['recentEvent'] = null;
      if (todayPnl > 0) event = 'profit';
      else if (todayPnl < -50) event = 'loss';
      else if (openPositions > 0) event = 'hold';

      let emotion: RobotState['emotion'] = 'idle';
      let gaze: RobotState['gaze'] = 'center';

      if (dailyLoss < -150 || !safetyNet) {
        emotion = 'worried'; gaze = 'down';
      } else if (todayPnl > 100) {
        emotion = 'excited'; gaze = 'up';
      } else if (openPositions > 2) {
        emotion = 'focused'; gaze = 'left';
      } else if (openPositions > 0) {
        emotion = 'thinking'; gaze = 'center';
      } else {
        emotion = 'idle'; gaze = 'center';
      }

      setRobotState({ emotion, gaze, recentEvent: event, pnl: todayPnl });

      const time = new Date().toLocaleTimeString('en-US', { hour12: false });
      setLogs(prev => {
        const newLogs = [`[${time}] ${openPositions} pos | P&L ${fmt$(todayPnl)} | Safety: ${safetyNet ? 'ON' : 'OFF'}`];
        return [...newLogs, ...prev].slice(0, 30);
      });
    } catch (e: any) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    fetchTelemetry();
    const id = setInterval(fetchTelemetry, 2000);
    return () => clearInterval(id);
  }, [fetchTelemetry]);

  const positions = data?.journal?.journal?.positions || [];
  const todayPnl = data?.journal?.journal?.today_pnl || 0;
  const totalPnl = data?.journal?.journal?.total_pnl || 0;
  const openCount = positions.length;
  const safetyNet = data?.risk?.risk?.safety_enabled ?? true;
  const regime = data?.brain?.consolidation?.regime ?? 'N/A';
  const halim = data?.brain?.consolidation?.halim ?? 'N/A';
  const thinker = data?.brain?.consolidation?.thinker ?? 'N/A';
  const connected = data?.health?.connected ?? false;

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-cyan-500/20 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-cyan-500/20 border border-cyan-400 flex items-center justify-center">
            <span className="text-xl">🤖</span>
          </div>
          <div>
            <h1 className="font-orbitron font-bold text-lg tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">
              JULI
            </h1>
            <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
              <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'}`} />
              <span>{connected ? 'CONNECTED' : 'DISCONNECTED'}</span>
            </div>
          </div>
        </div>

        <div className="hidden lg:flex items-center gap-6 font-mono text-xs">
          <div className="text-slate-400">
            SAFETY: <span className={safetyNet ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>{safetyNet ? 'ON' : 'OFF'}</span>
          </div>
          <div className="text-slate-400">
            POSITIONS: <span className="text-cyan-400 font-bold">{openCount}</span>
          </div>
          <div className="text-slate-400">
            TODAY: <span className={todayPnl >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>{fmt$(todayPnl)}</span>
          </div>
        </div>
      </header>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-0 overflow-hidden">
        <div className="lg:col-span-2 flex flex-col overflow-hidden">
          <div className="relative flex-1 min-h-[400px]">
            <JuliRobot state={robotState} />
          </div>
          <div className="border-t border-cyan-500/20 p-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
            <MetricCard label="TODAY P&L" value={fmt$(todayPnl)} positive={todayPnl >= 0} />
            <MetricCard label="TOTAL P&L" value={fmt$(totalPnl)} positive={totalPnl >= 0} />
            <MetricCard label="REGIME" value={typeof regime === 'number' ? regime.toFixed(2) : String(regime)} neutral />
            <MetricCard label="THINKER" value={typeof thinker === 'number' ? thinker.toFixed(4) : String(thinker)} neutral />
          </div>
        </div>

        <div className="border-l border-cyan-500/20 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-auto p-3 border-b border-cyan-500/20">
            <h3 className="font-orbitron text-xs font-bold text-cyan-400 tracking-wider mb-2">OPEN POSITIONS</h3>
            {positions.length === 0 ? (
              <p className="text-xs text-slate-500 font-mono">No open positions</p>
            ) : (
              <div className="space-y-1.5">
                {positions.map((p: any, i: number) => (
                  <div key={i} className="flex items-center justify-between text-xs font-mono bg-slate-900/50 rounded-lg px-2 py-1.5 border border-slate-800">
                    <span className="text-cyan-300 font-bold">{p.symbol || p.ticker || '?'}</span>
                    <span className={p.side === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}>
                      {p.side || p.action || '?'}
                    </span>
                    <span className="text-slate-400">{p.qty || p.quantity || '?'}</span>
                    <span className={(p.pnl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {fmt$(p.pnl || 0)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="p-3 border-b border-cyan-500/20">
            <h3 className="font-orbitron text-xs font-bold text-cyan-400 tracking-wider mb-2">JULI BRAIN</h3>
            <div className="grid grid-cols-3 gap-2 text-center">
              <MiniStat label="REGIME" value={typeof regime === 'number' ? regime.toFixed(2) : String(regime)} />
              <MiniStat label="HALIM" value={typeof halim === 'number' ? halim.toFixed(3) : String(halim)} />
              <MiniStat label="THINKER" value={typeof thinker === 'number' ? thinker.toFixed(4) : String(thinker)} />
            </div>
          </div>

          <div className="flex-1 overflow-auto p-3">
            <h3 className="font-orbitron text-xs font-bold text-cyan-400 tracking-wider mb-2">LIVE LOG</h3>
            <div className="space-y-0.5 font-mono text-[10px] text-slate-400">
              {logs.map((log, i) => (
                <div key={i} className={i === 0 ? 'text-cyan-300' : ''}>{log}</div>
              ))}
              {logs.length === 0 && <div className="text-slate-600">Waiting for data...</div>}
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="fixed bottom-4 right-4 bg-red-950/90 border border-red-500/50 rounded-lg px-3 py-2 text-xs text-red-300 font-mono">
          {error}
        </div>
      )}
    </div>
  );
}

function MetricCard({ label, value, positive, neutral }: { label: string; value: string; positive?: boolean; neutral?: boolean }) {
  return (
    <div className="bg-slate-900/50 rounded-xl border border-slate-800 p-2">
      <div className="text-[10px] font-mono text-slate-500 tracking-wider">{label}</div>
      <div className={`font-orbitron font-bold text-sm mt-0.5 ${neutral ? 'text-cyan-400' : positive ? 'text-emerald-400' : 'text-rose-400'}`}>
        {value}
      </div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[9px] font-mono text-slate-600">{label}</div>
      <div className="text-xs font-orbitron font-bold text-cyan-300">{value}</div>
    </div>
  );
}
