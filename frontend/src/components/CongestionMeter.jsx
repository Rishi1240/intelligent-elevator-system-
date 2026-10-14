import React from 'react';

/**
 * CongestionMeter Component
 * Visualizes predicted congestion level from LSTM on a 0-100% scale.
 * Shows the 40% Demand-Aware Predictive Positioning threshold,
 * per-floor demand predictions (compact heatmap bar chart),
 * highest-demand floor highlight, and active controller status with allocations.
 */
export default function CongestionMeter({
  congestionLevel = 0.0,
  horizonMin = 5,
  targetFloor = null,
  targetDemand = 0.0,
  positioningStatus = 'inactive',
  assignedElevators = [],
  floorDemandT5 = [],
  assignedAllocations = [],
  eligibleElevatorsCount = 0,
  lastDecisionLog = '',
  highThreshold = 0.55,
  numFloors = 10,
}) {
  const percentage = Math.min(100, Math.max(0, Math.round(congestionLevel * 100)));
  const thresholdPct = Math.round(highThreshold * 100);

  let barColor = '#10b981'; // Green (< 25%)
  let label = 'Low Congestion (Normal)';
  if (percentage >= thresholdPct) {
    barColor = '#ef4444'; // Red (>= threshold)
    label = `High Congestion (≥${thresholdPct}% Peak Rush)`;
  } else if (percentage >= 25) {
    barColor = '#f59e0b'; // Yellow (25-threshold)
    label = 'Moderate Traffic';
  }

  // Positioning status badge formatting
  const statusStyles = {
    active: { bg: '#450a0a', border: '#ef4444', text: '#fca5a5', icon: '⚡', label: 'PRE-POSITIONING ACTIVE' },
    awaiting_confirmation: { bg: '#3b2504', border: '#f59e0b', text: '#fde68a', icon: '⏳', label: 'AWAITING CONFIRMATION' },
    cooling_down: { bg: '#1e1b4b', border: '#818cf8', text: '#c7d2fe', icon: '❄️', label: 'COOLING DOWN' },
    unavailable: { bg: '#262626', border: '#737373', text: '#d4d4d4', icon: '⚠️', label: 'MODEL UNAVAILABLE' },
    inactive: { bg: '#064e3b', border: '#10b981', text: '#6ee7b7', icon: '✓', label: 'NORMAL OPERATION' },
  };

  const currentStatus = statusStyles[positioningStatus] || statusStyles.inactive;

  // Floor demand analysis
  const demands = floorDemandT5.length > 0 ? floorDemandT5 : [];
  const maxDemand = demands.length > 0 ? Math.max(...demands) : 0;
  const peakFloorIdx = demands.length > 0 ? demands.indexOf(maxDemand) : -1;
  const peakFloor = peakFloorIdx >= 0 ? peakFloorIdx + 1 : null;

  // Build allocation map: floor -> [elevator_ids]
  const allocationMap = {};
  if (assignedAllocations && assignedAllocations.length > 0) {
    assignedAllocations.forEach((a) => {
      const f = a.target_floor;
      if (!allocationMap[f]) allocationMap[f] = [];
      allocationMap[f].push(a.elevator_id + 1);
    });
  }

  // Color for demand bar based on demand level
  const getDemandColor = (demand, isPeak) => {
    if (isPeak && demand > 0) return '#ef4444';
    if (demand >= 4) return '#f97316';
    if (demand >= 2) return '#f59e0b';
    if (demand >= 1) return '#22d3ee';
    return '#334155';
  };

  return (
    <div
      style={{
        backgroundColor: '#1e293b',
        padding: '16px 20px',
        borderRadius: '12px',
        border: '1px solid #334155',
        marginBottom: '20px',
      }}
    >
      {/* Header Row */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '10px',
          flexWrap: 'wrap',
          gap: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: '#94a3b8', fontSize: '13px', fontWeight: '700', textTransform: 'uppercase' }}>
            LSTM Traffic Congestion Forecast (+{horizonMin} min)
          </span>
          {peakFloor && (
            <span
              style={{
                fontSize: '11px',
                fontWeight: '700',
                color: '#38bdf8',
                backgroundColor: '#0f172a',
                padding: '2px 8px',
                borderRadius: '6px',
                border: '1px solid #0284c7',
              }}
            >
              Peak Demand: Floor {peakFloor} ({maxDemand.toFixed(2)} reqs)
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Status Badge */}
          <span
            style={{
              fontSize: '11px',
              fontWeight: '800',
              padding: '3px 8px',
              borderRadius: '8px',
              backgroundColor: currentStatus.bg,
              border: `1px solid ${currentStatus.border}`,
              color: currentStatus.text,
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <span>{currentStatus.icon}</span>
            <span>{currentStatus.label}</span>
            {positioningStatus === 'active' && assignedElevators.length > 0 && (
              <span>({assignedElevators.map((e) => `L${e + 1}`).join(', ')})</span>
            )}
          </span>

          <span
            style={{
              fontSize: '14px',
              fontWeight: 'bold',
              color: barColor,
            }}
          >
            {percentage}% — {label}
          </span>
        </div>
      </div>

      {/* Progress Bar Container with Threshold Line */}
      <div
        style={{
          position: 'relative',
          height: '16px',
          width: '100%',
          backgroundColor: '#0f172a',
          borderRadius: '8px',
          overflow: 'hidden',
          border: '1px solid #334155',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${percentage}%`,
            backgroundColor: barColor,
            borderRadius: '8px',
            transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.5s ease',
            boxShadow: `0 0 12px ${barColor}88`,
          }}
        />

        {/* Threshold Vertical Marker Line */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: `${thresholdPct}%`,
            width: '2px',
            backgroundColor: '#ffffff',
            boxShadow: '0 0 4px #ffffff',
            zIndex: 2,
          }}
          title={`${thresholdPct}% Pre-Positioning Activation Threshold`}
        />
      </div>

      {/* Threshold Labels Below Bar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: '6px',
          fontSize: '10px',
          color: '#64748b',
          fontWeight: '600',
        }}
      >
        <span>0% (Idle)</span>
        <span>30% (Moderate)</span>
        <span style={{ color: '#ef4444', fontWeight: '700' }}>▲ {thresholdPct}% Pre-Positioning Trigger</span>
        <span>100% (Saturated)</span>
      </div>

      {/* ── Floor-Wise Demand Forecast ──────────────────────────── */}
      {demands.length > 0 && (
        <div style={{ marginTop: '14px' }}>
          <div style={{ color: '#94a3b8', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', marginBottom: '8px' }}>
            Floor-Wise Predicted Demand (+{horizonMin} min)
          </div>

          {/* Compact bar chart */}
          <div style={{ display: 'flex', gap: '3px', alignItems: 'flex-end', height: '60px' }}>
            {demands.map((dem, idx) => {
              const isPeak = idx === peakFloorIdx && dem > 0;
              const barHeight = maxDemand > 0 ? Math.max(4, (dem / maxDemand) * 56) : 4;
              const floorNum = idx + 1;
              const allocatedLifts = allocationMap[floorNum] || [];

              return (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    flex: 1,
                    position: 'relative',
                  }}
                  title={`F${floorNum}: ${dem.toFixed(2)} reqs${isPeak ? ' (PEAK)' : ''}${allocatedLifts.length > 0 ? ` [Lift ${allocatedLifts.join(',')}]` : ''}`}
                >
                  {/* Demand value */}
                  <span style={{
                    fontSize: '9px',
                    color: isPeak ? '#fca5a5' : '#94a3b8',
                    fontWeight: isPeak ? '800' : '500',
                    marginBottom: '2px',
                  }}>
                    {dem.toFixed(1)}
                  </span>
                  {/* Bar */}
                  <div
                    style={{
                      width: '100%',
                      height: `${barHeight}px`,
                      backgroundColor: getDemandColor(dem, isPeak),
                      borderRadius: '3px 3px 0 0',
                      transition: 'height 0.4s ease, background-color 0.3s ease',
                      border: isPeak ? '1px solid #ef4444' : 'none',
                      boxShadow: isPeak ? '0 0 8px #ef444488' : 'none',
                    }}
                  />
                  {/* Floor label */}
                  <span style={{
                    fontSize: '9px',
                    color: isPeak ? '#f87171' : '#64748b',
                    fontWeight: isPeak ? '800' : '600',
                    marginTop: '2px',
                  }}>
                    F{floorNum}
                  </span>
                  {/* Allocated elevator indicator */}
                  {allocatedLifts.length > 0 && (
                    <span style={{
                      fontSize: '8px',
                      color: '#818cf8',
                      fontWeight: '700',
                      marginTop: '1px',
                    }}>
                      {allocatedLifts.map(l => `L${l}`).join(',')}
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          {/* Allocation Summary Row */}
          {positioningStatus === 'active' && assignedAllocations && assignedAllocations.length > 0 && (
            <div style={{
              marginTop: '10px',
              padding: '8px 12px',
              backgroundColor: '#0f172a',
              borderRadius: '8px',
              border: '1px solid #1e3a5f',
            }}>
              <div style={{ color: '#94a3b8', fontSize: '10px', fontWeight: '700', textTransform: 'uppercase', marginBottom: '5px' }}>
                Predictive Allocations ({eligibleElevatorsCount} eligible elevator{eligibleElevatorsCount !== 1 ? 's' : ''})
              </div>
              {assignedAllocations.map((alloc, i) => (
                <div key={i} style={{
                  fontSize: '11px',
                  color: '#e2e8f0',
                  padding: '2px 0',
                  display: 'flex',
                  gap: '8px',
                  alignItems: 'center',
                }}>
                  <span style={{
                    backgroundColor: '#312e81',
                    color: '#a5b4fc',
                    padding: '1px 6px',
                    borderRadius: '4px',
                    fontSize: '10px',
                    fontWeight: '700',
                  }}>
                    Lift {alloc.elevator_id + 1}
                  </span>
                  <span>→</span>
                  <span style={{ fontWeight: '700', color: '#38bdf8' }}>F{alloc.target_floor}</span>
                  <span style={{ color: '#64748b', fontSize: '10px' }}>— {alloc.reason}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
