import React from 'react';

/**
 * ElevatorShaft Component
 * Renders a single vertical elevator shaft with numFloors stacked floors
 * (top floor at the top), dynamic car indicator highlighting currentFloor,
 * and occupancy telemetry.
 */
export default function ElevatorShaft({
  elevatorId,
  currentFloor,
  occupancy = 0,
  capacity = 8,
  numFloors = 10,
  assignedStops = [],
  targetFloor = null,
}) {
  // Generate floors top → bottom
  const floors = Array.from({ length: numFloors }, (_, i) => numFloors - i);

  // Scale shaft height based on number of floors (40px per floor, min 200px)
  const floorHeight = Math.max(28, Math.min(42, Math.floor(420 / numFloors)));

  const isStop = (f) => assignedStops.includes(f);
  const isTarget = targetFloor === currentFloor;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        margin: '0 6px',
        userSelect: 'none',
      }}
    >
      {/* Elevator ID Header */}
      <div
        style={{
          color: '#00d4ff',
          fontWeight: '700',
          fontSize: '13px',
          marginBottom: '6px',
          letterSpacing: '0.5px',
          textTransform: 'uppercase',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
        }}
      >
        <span>Lift {elevatorId}</span>
        {assignedStops.length > 0 && (
          <span
            style={{
              fontSize: '9px',
              padding: '1px 5px',
              borderRadius: '4px',
              backgroundColor: '#a855f7',
              color: '#fff',
              fontWeight: '800',
            }}
          >
            {assignedStops[0] > currentFloor ? '▲' : '▼'}
          </span>
        )}
      </div>

      {/* Shaft Body */}
      <div
        style={{
          width: '78px',
          backgroundColor: '#0f172a',
          borderRadius: '8px',
          border: '2px solid #334155',
          overflow: 'hidden',
          boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.6), 0 4px 12px rgba(0,0,0,0.3)',
        }}
      >
        {floors.map((floorNum) => {
          const isAtCurrentFloor = floorNum === currentFloor;
          const isAssigned = isStop(floorNum);
          return (
            <div
              key={floorNum}
              style={{
                height: `${floorHeight}px`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0 6px',
                borderBottom: floorNum === 1 ? 'none' : '1px solid #1e293b',
                backgroundColor: isAtCurrentFloor
                  ? '#00d4ff'
                  : isAssigned
                  ? 'rgba(168, 85, 247, 0.15)'
                  : 'transparent',
                color: isAtCurrentFloor
                  ? '#0a0f1d'
                  : isAssigned
                  ? '#c084fc'
                  : '#64748b',
                fontWeight: isAtCurrentFloor ? '800' : isAssigned ? '700' : '500',
                fontSize: '11px',
                transition: 'all 0.3s ease',
                boxShadow: isAtCurrentFloor
                  ? '0 0 16px rgba(0, 212, 255, 0.8), inset 0 0 6px rgba(255,255,255,0.4)'
                  : 'none',
              }}
            >
              <span>F{floorNum}</span>
              {isAtCurrentFloor ? (
                <span
                  style={{
                    fontSize: '9px',
                    backgroundColor: '#0a0f1d',
                    color: '#00d4ff',
                    padding: '1px 4px',
                    borderRadius: '3px',
                    fontWeight: 'bold',
                  }}
                >
                  CAR
                </span>
              ) : isAssigned ? (
                <span
                  style={{
                    fontSize: '8px',
                    backgroundColor: '#7e22ce',
                    color: '#f3e8ff',
                    padding: '1px 3px',
                    borderRadius: '3px',
                    fontWeight: '700',
                  }}
                >
                  STOP
                </span>
              ) : (
                <span style={{ fontSize: '9px', opacity: 0.3 }}>|</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Occupancy Indicator */}
      <div
        style={{
          marginTop: '8px',
          fontSize: '11px',
          color: occupancy >= capacity ? '#ef4444' : '#94a3b8',
          fontWeight: '600',
          textAlign: 'center',
          backgroundColor: '#1e293b',
          padding: '3px 7px',
          borderRadius: '6px',
          border: '1px solid #334155',
          width: '78px',
        }}
      >
        {occupancy}/{capacity} pax
      </div>

      {/* Dynamic Route / Stops Queue */}
      <div
        style={{
          marginTop: '4px',
          fontSize: '9px',
          color: assignedStops.length > 0 ? '#c084fc' : '#475569',
          fontWeight: '700',
          textAlign: 'center',
          backgroundColor: assignedStops.length > 0 ? 'rgba(147, 51, 234, 0.15)' : '#0f172a',
          padding: '2px 5px',
          borderRadius: '4px',
          border: `1px solid ${assignedStops.length > 0 ? '#7e22ce' : '#1e293b'}`,
          maxWidth: '92px',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
        title={assignedStops.length > 0 ? `Stops: ${assignedStops.map(s => `F${s}`).join(' → ')}` : 'Stationary Idle'}
      >
        {assignedStops.length > 0
          ? assignedStops.map(s => `F${s}`).join('→')
          : 'IDLE'}
      </div>
    </div>
  );
}
