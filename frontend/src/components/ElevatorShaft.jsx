import React from 'react';

/**
 * ElevatorShaft Component
 * Renders a single vertical elevator shaft with numFloors stacked floors
 * (top floor at the top), dynamic car indicator highlighting currentFloor,
 * occupancy telemetry, explicit emergency indicators, and predictive pre-positioning status.
 */
export default function ElevatorShaft({
  elevatorId,
  currentFloor,
  occupancy = 0,
  capacity = 8,
  numFloors = 10,
  assignedStops = [],
  targetFloor = null,
  emergencyInfo = null,
  operationalStatus = 'idle',
  predictiveTarget = null,
}) {
  // Generate floors top → bottom
  const floors = Array.from({ length: numFloors }, (_, i) => numFloors - i);

  // Scale shaft height based on number of floors (40px per floor, min 200px)
  const floorHeight = Math.max(28, Math.min(42, Math.floor(420 / numFloors)));

  const isStop = (f) => assignedStops.includes(f);

  // Emergency status derivation
  const isEmergency = Boolean(
    emergencyInfo &&
    ['en_route', 'dispatched', 'arrived', 'assigned'].includes(emergencyInfo.status)
  );
  const isEmergencyArrived = isEmergency && (
    emergencyInfo.status === 'arrived' ||
    currentFloor === emergencyInfo.floor
  );
  const emergencyTargetFloor = emergencyInfo?.floor;

  const isRepositioning = operationalStatus === 'repositioning' || (predictiveTarget !== null && predictiveTarget !== undefined);
  const isServingPassenger = operationalStatus === 'serving_passenger' || occupancy > 0;

  // Header badge calculation
  let headerColor = '#00d4ff';
  let headerBadge = null;

  if (isEmergency) {
    headerColor = '#f87171';
    headerBadge = (
      <span
        style={{
          fontSize: '9px',
          padding: '1px 5px',
          borderRadius: '4px',
          backgroundColor: isEmergencyArrived ? '#059669' : '#dc2626',
          color: '#fff',
          fontWeight: '800',
          boxShadow: isEmergencyArrived
            ? '0 0 8px rgba(16, 185, 129, 0.6)'
            : '0 0 8px rgba(239, 68, 68, 0.6)',
        }}
      >
        {isEmergencyArrived ? '✓ ARRIVED' : emergencyTargetFloor > currentFloor ? '▲ EMG' : '▼ EMG'}
      </span>
    );
  } else if (isRepositioning) {
    headerColor = '#fb923c';
    const effectiveTarget = predictiveTarget || targetFloor;
    const direction = effectiveTarget > currentFloor ? '▲' : '▼';
    headerBadge = (
      <span
        style={{
          fontSize: '9px',
          padding: '1px 5px',
          borderRadius: '4px',
          backgroundColor: '#ea580c',
          color: '#fff',
          fontWeight: '800',
        }}
      >
        {direction} PRE-POS
      </span>
    );
  } else if (assignedStops.length > 0) {
    headerColor = '#c084fc';
    headerBadge = (
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
    );
  }

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
          color: headerColor,
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
        {headerBadge}
      </div>

      {/* Shaft Body */}
      <div
        style={{
          width: '78px',
          backgroundColor: '#0f172a',
          borderRadius: '8px',
          border: `2px solid ${
            isEmergency
              ? isEmergencyArrived ? '#10b981' : '#ef4444'
              : isRepositioning
              ? '#f97316'
              : '#334155'
          }`,
          overflow: 'hidden',
          boxShadow: isEmergency
            ? `inset 0 2px 8px rgba(0,0,0,0.6), 0 0 12px ${isEmergencyArrived ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`
            : isRepositioning
            ? 'inset 0 2px 8px rgba(0,0,0,0.6), 0 0 10px rgba(249, 115, 22, 0.4)'
            : 'inset 0 2px 8px rgba(0,0,0,0.6), 0 4px 12px rgba(0,0,0,0.3)',
        }}
      >
        {floors.map((floorNum) => {
          const isAtCurrentFloor = floorNum === currentFloor;
          const isAssigned = isStop(floorNum);
          const isEmergencyFloor = isEmergency && floorNum === emergencyTargetFloor;
          const isPredictiveFloor = isRepositioning && floorNum === (predictiveTarget || targetFloor);

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
                  ? isEmergency ? '#ef4444' : isRepositioning ? '#ea580c' : '#00d4ff'
                  : isEmergencyFloor
                  ? 'rgba(239, 68, 68, 0.25)'
                  : isPredictiveFloor
                  ? 'rgba(249, 115, 22, 0.20)'
                  : isAssigned
                  ? 'rgba(168, 85, 247, 0.15)'
                  : 'transparent',
                color: isAtCurrentFloor
                  ? '#0a0f1d'
                  : isEmergencyFloor
                  ? '#fca5a5'
                  : isPredictiveFloor
                  ? '#fdba74'
                  : isAssigned
                  ? '#c084fc'
                  : '#64748b',
                fontWeight: isAtCurrentFloor ? '800' : isEmergencyFloor || isAssigned || isPredictiveFloor ? '700' : '500',
                fontSize: '11px',
                transition: 'all 0.3s ease',
                boxShadow: isAtCurrentFloor
                  ? isEmergency
                    ? '0 0 16px rgba(239, 68, 68, 0.9), inset 0 0 6px rgba(255,255,255,0.5)'
                    : isRepositioning
                    ? '0 0 16px rgba(234, 88, 12, 0.9), inset 0 0 6px rgba(255,255,255,0.5)'
                    : '0 0 16px rgba(0, 212, 255, 0.8), inset 0 0 6px rgba(255,255,255,0.4)'
                  : 'none',
              }}
            >
              <span>F{floorNum}</span>
              {isAtCurrentFloor ? (
                <span
                  style={{
                    fontSize: '9px',
                    backgroundColor: '#0a0f1d',
                    color: isEmergency ? '#ef4444' : isRepositioning ? '#ea580c' : '#00d4ff',
                    padding: '1px 4px',
                    borderRadius: '3px',
                    fontWeight: 'bold',
                  }}
                >
                  {isEmergencyArrived ? 'EMG' : isRepositioning ? 'POS' : 'CAR'}
                </span>
              ) : isEmergencyFloor ? (
                <span
                  style={{
                    fontSize: '8px',
                    backgroundColor: '#b91c1c',
                    color: '#fee2e2',
                    padding: '1px 3px',
                    borderRadius: '3px',
                    fontWeight: '800',
                  }}
                >
                  🚨EMG
                </span>
              ) : isPredictiveFloor ? (
                <span
                  style={{
                    fontSize: '8px',
                    backgroundColor: '#c2410c',
                    color: '#ffedd5',
                    padding: '1px 3px',
                    borderRadius: '3px',
                    fontWeight: '800',
                  }}
                >
                  🎯PRE
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

      {/* Operational Status Badge */}
      <div
        style={{
          marginTop: '4px',
          fontSize: '9px',
          color: isEmergency
            ? isEmergencyArrived ? '#6ee7b7' : '#fca5a5'
            : isRepositioning
            ? '#fdba74'
            : isServingPassenger
            ? '#c084fc'
            : '#64748b',
          fontWeight: '700',
          textAlign: 'center',
          backgroundColor: isEmergency
            ? isEmergencyArrived ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)'
            : isRepositioning
            ? 'rgba(234, 88, 12, 0.2)'
            : isServingPassenger
            ? 'rgba(147, 51, 234, 0.15)'
            : '#0f172a',
          padding: '3px 6px',
          borderRadius: '4px',
          border: `1px solid ${
            isEmergency
              ? isEmergencyArrived ? '#10b981' : '#ef4444'
              : isRepositioning
              ? '#ea580c'
              : isServingPassenger
              ? '#7e22ce'
              : '#1e293b'
          }`,
          maxWidth: '92px',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
        title={
          isEmergency
            ? isEmergencyArrived
              ? `Emergency Arrived at Floor ${emergencyTargetFloor}`
              : `En Route to Emergency at Floor ${emergencyTargetFloor}`
            : isRepositioning
            ? `Predictive Pre-Positioning to F${predictiveTarget || targetFloor}`
            : isServingPassenger
            ? `Serving Passengers: Stops ${assignedStops.map((s) => `F${s}`).join(' → ')}`
            : 'Stationary Idle'
        }
      >
        {isEmergency
          ? isEmergencyArrived
            ? `🚨 AT F${emergencyTargetFloor}`
            : `🚨 TO F${emergencyTargetFloor}`
          : isRepositioning
          ? `⚡ TO F${predictiveTarget || targetFloor}`
          : assignedStops.length > 0
          ? assignedStops.map((s) => `F${s}`).join('→')
          : 'IDLE'}
      </div>
    </div>
  );
}
