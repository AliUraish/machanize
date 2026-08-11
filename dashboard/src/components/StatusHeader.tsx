import type { Health } from "../types";

type Props = {
  health: Health | null;
};

export function StatusHeader({ health }: Props) {
  return (
    <header className="status-header">
      <div>
        <div className="eyebrow">MACHANIZE / PHASE 3</div>
        <h1>Visual training review</h1>
      </div>
      <div className="safety-badge" role="status">
        <span className="safety-dot" />
        ROBOT MOVEMENT {health?.robot_movement_enabled === false ? "DISABLED" : "CHECKING"}
      </div>
    </header>
  );
}
